from scalebridge.dashapp.services.system import (
    PACKAGE_GROUPS,path_snapshot,machine_snapshot,environment_variables,
    package_snapshot,mlflow_snapshot,
)
def test_data_scalebridge_contract():
    assert path_snapshot()["roots"]["ScaleBridge Data Root"].replace("\\","/").endswith("/Data/ScaleBridge")
def test_current_machine_and_python_are_reported():
    s=machine_snapshot(); assert s["Hostname"] and s["Python Executable"]
def test_known_variables_report_values_or_not_set():
    rows=environment_variables(); names={r["name"] for r in rows}
    assert "SCALEBRIDGE_MACHINE_ID" in names and "MLFLOW_TRACKING_URI" in names
    assert all(r["value"] for r in rows)
def test_complete_package_scope_is_present():
    s=package_snapshot(); assert len(s)==len(PACKAGE_GROUPS)
    names={r["component"] for g in s for r in g["packages"]}
    for n in ("numpy","dash","dash-bootstrap-components","mlflow","torch","casadi","opyplus","gymnasium","opendssdirect","scalebridge"):
        assert n in names
def test_mlflow_missing_values_are_explicit():
    s=mlflow_snapshot(); assert s["MLFLOW_TRACKING_URI"] and s["MLFLOW_BACKEND_STORE_URI"]
