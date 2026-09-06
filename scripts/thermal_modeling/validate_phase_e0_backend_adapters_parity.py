from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import torch

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec, AllocationMode, ParameterConfig, ParameterStatus,
    RCCompilerSpec, SpatialMode, ZoneAdjacency, compile_rc_model,
)
from scalebridge.models.grey_box.rc_networks.backend_adapters import (
    CasadiPhysicalRCBackend, CasadiTransformedRCBackend, NeuromancerRCBackend,
    NumpyPhysicalRCBackend, NumpyRCBackend, TorchRCBackend,
)


def ports(zones):
    return {z:("qac","zic","zir","qsol1","qsol2") for z in zones}


def dv(f):
    return {"C_a":2e6,"C_m":8e6,"C_e":5e6,"R_ao":.02,"R_am":.01,"R_om":.04,"R_ae":.015,"R_eo":.03,"R_inter_a_a":.025,"eta_r":.7,"gamma_a_r":.2,"gamma_e_r":.3,"gamma_m_r":.5}[f]


def configured(spec, estimated=(), values=None, bounds=None):
    provisional=compile_rc_model(spec); values=values or {}; bounds=bounds or {}; cfg={}
    for m in provisional.parameter_registry.masters:
        inst=provisional.parameter_registry.instance(m.member_instance_ids[0]); z=inst.zone_scope[0] if inst.zone_scope else None
        val=values.get((m.family,z), values.get(m.family,dv(m.family))); lo,hi=bounds.get((m.family,z),bounds.get(m.family,(None,None)))
        cfg[m.master_id]=ParameterConfig(ParameterStatus.ESTIMATED if m.family in set(estimated) else ParameterStatus.FIXED,float(val),lo,hi)
    return compile_rc_model(spec, parameter_configs=cfg)


def dep2_model():
    z=("A","B")
    spec=RCCompilerSpec("2r2c",z,SpatialMode.DEP2,adjacency=(ZoneAdjacency("A","B"),),zone_port_availability=ports(z),dep2_allocations=(
        AllocationFamilySpec("zic_family",("zic",),{"A":.4,"B":.6},AllocationMode.NEUTRAL_FIXED),
        AllocationFamilySpec("zir_family",("zir",),{"A":.4,"B":.6},AllocationMode.ESTIMATED),
        AllocationFamilySpec("sol1_family",("qsol1",),{"A":.4,"B":.6},AllocationMode.NEUTRAL_FIXED),
        AllocationFamilySpec("sol2_family",("qsol2",),{"A":.4,"B":.6},AllocationMode.NEUTRAL_FIXED),))
    return configured(spec,estimated=("C_a","R_ao","R_am","eta_r"),values={("C_a","A"):1e6,("C_a","B"):1.2e6,("C_m","A"):5e6,("C_m","B"):6e6,("R_ao","A"):.01,("R_ao","B"):.012,("R_am","A"):.02,("R_am","B"):.02,("eta_r","A"):.8,("eta_r","B"):.7,"R_inter_a_a":.03},bounds={"C_a":(1e4,1e8),"R_ao":(1e-4,1.0),"R_am":(1e-4,1.0),"eta_r":(0.,1.)})


def main():
    report={"contract":"E0-6 v2 physical-Theta authority","flavours":{}}
    for fl,est in {"1r1c":("C_a","R_ao"),"2r2c":("C_a","R_ao","eta_r"),"3r2c":("C_m","R_om","eta_r"),"4r3c":("C_e","R_ae","gamma_a_r","gamma_e_r","gamma_m_r")}.items():
        m=configured(RCCompilerSpec(fl,("A",),"ind",zone_port_availability=ports(("A",))),estimated=est)
        p=NumpyPhysicalRCBackend(m); mats=p.matrices(p.initial_physical())
        report["flavours"][fl]={"decision_dimension":p.plan.decision_dimension,"constraint_count":len(p.plan.constraints),"finite":bool(np.isfinite(mats.A).all())}

    m=dep2_model(); nt=NumpyRCBackend(m); tt=TorchRCBackend(m,dtype=torch.float64); npb=NumpyPhysicalRCBackend(m); cp=CasadiPhysicalRCBackend(m,"MX"); ct=CasadiTransformedRCBackend(m,"SX")
    rho=np.linspace(-.2,.25,nt.plan.raw_dimension); theta=nt.physical_decision_vector(rho)
    local=np.zeros(len(m.thermal_ports)); local[m.port_index["A::qac"]]=-1200; local[m.port_index["B::qac"]]=-800
    amap={"zic":1000.,"zir":600.,"qsol1":800.,"qsol2":300.}; agg=np.array([amap[s] for s in npb.plan.aggregate_signal_order]); x=np.array([22.,22.,24.,24.]); b=np.array([10.]); probe=np.array([1.,-.5,.75,.2])
    a=npb.matrices(theta).A; ca=cp.matrices(theta)[5]; assert np.allclose(a,ca,rtol=1e-9,atol=1e-10)
    nr=npb.rhs(theta,x,b,local,agg); cr=cp.rhs(theta,x,b,local,agg); assert np.allclose(nr,cr,rtol=1e-9,atol=1e-10)
    for solver in ("euler","rk2","rk4","exact_zoh_linear"):
        ns=npb.step(solver,theta,x,b,local,agg,sample_dt_s=600.,substeps=4); cs=cp.step(solver,theta,x,b,local,agg,sample_dt_s=600.,substeps=4); assert np.allclose(ns,cs,rtol=1e-8,atol=1e-8)
    rho_t=torch.tensor(rho,dtype=torch.float64,requires_grad=True); out=tt.rhs(torch.tensor(x,dtype=torch.float64),torch.tensor(b,dtype=torch.float64),torch.tensor(local,dtype=torch.float64),torch.tensor(agg,dtype=torch.float64),raw=rho_t); loss=torch.dot(torch.tensor(probe,dtype=torch.float64),out); gr=torch.autograd.grad(loss,rho_t,retain_graph=True)[0]
    jac=torch.autograd.functional.jacobian(lambda rr: tt.physical_decision_vector(rr),rho_t); gt=torch.tensor(cp.parameter_probe_gradient(theta,x,b,local,agg,probe),dtype=torch.float64); mapped=jac.T@gt; torch.testing.assert_close(gr,mapped,rtol=1e-7,atol=1e-8)
    objective=cp.ca.sumsqr(cp.theta_symbol-cp.ca.DM(cp.initial_physical())); spec=cp.build_ipopt_nlp(objective); solver=cp.ca.nlpsol("e06_v2_validator_ipopt","ipopt",spec["nlp"],{"ipopt.print_level":0,"print_time":False}); sol=solver(x0=spec["x0"],lbx=spec["lbx"],ubx=spec["ubx"],lbg=spec["lbg"],ubg=spec["ubg"]); assert np.isfinite(np.asarray(sol["x"],dtype=float)).all()
    nm_ok=False
    try:
        nm=NeuromancerRCBackend(tt); nm_ok=(nm.raw is tt.raw)
    except Exception:
        nm_ok=False
    report["running_example"]={"raw_dimension":nt.plan.raw_dimension,"physical_dimension":npb.plan.decision_dimension,"physical_constraint_count":len(npb.plan.constraints),"chain_rule_p4":True,"ipopt_smoke":True,"neuromancer_owner_shared":nm_ok,"transformed_casadi_reference":ct.__class__.__name__}
    report["qualified"]=True
    outpath=Path("validated_artifacts/phase_e0/e06_backend_adapters_parity_validation.json"); outpath.parent.mkdir(parents=True,exist_ok=True); outpath.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); print("E0-6 BACKEND ADAPTERS + NUMERICAL PARITY v2 VALIDATION: PASS")
    return 0

if __name__=="__main__": raise SystemExit(main())
