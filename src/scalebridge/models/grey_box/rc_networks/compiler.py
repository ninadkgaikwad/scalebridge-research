from __future__ import annotations

"""Graph compiler for the frozen Phase E0 continuous-time RC mathematics."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .allocation import validate_allocation_spec
from .flavours import DEFAULT_CORE_PORT_GROUPS, RCFlavor, get_flavour
from .parameters import (
    ParameterConfig,
    ParameterInstance,
    ParameterRegistry,
    build_parameter_registry,
    make_capacitance_instance,
    make_resistance_instance,
    make_routing_instance,
)
from .specification import (
    AllocationFamilySpec,
    BoundaryNode,
    ConnectionRule,
    HeatPortGroup,
    RCCompileError,
    RCCompilerSpec,
    SpatialMode,
    StateNode,
    ThermalPort,
    ZoneAdjacency,
)


@dataclass(frozen=True)
class ResistanceEdge:
    edge_id: str
    node_a: StateNode | BoundaryNode
    node_b: StateNode | BoundaryNode
    family: str
    parameter_instance_id: str
    kind: str  # elemental_state, elemental_boundary, inter_zone

    def endpoint_keys(self) -> tuple[str, str]:
        return tuple(sorted((self.node_a.key, self.node_b.key)))  # type: ignore[return-value]


@dataclass(frozen=True)
class RCMatrices:
    """Numerical realization of the compiled continuous-time RC graph."""

    C: np.ndarray
    D: np.ndarray
    G: np.ndarray
    L: np.ndarray
    L_CC: np.ndarray
    L_CB: np.ndarray
    Gamma: np.ndarray
    H: np.ndarray
    instance_values: Mapping[str, float]


@dataclass(frozen=True)
class CompiledRCModel:
    spec: RCCompilerSpec
    flavour: RCFlavor
    state_nodes: tuple[StateNode, ...]
    boundary_nodes: tuple[BoundaryNode, ...]
    resistance_edges: tuple[ResistanceEdge, ...]
    thermal_ports: tuple[ThermalPort, ...]
    observed_nodes: tuple[StateNode, ...]
    state_capacitance_parameter: Mapping[str, str]
    routing_parameter_ids: Mapping[tuple[str, str], str]
    parameter_registry: ParameterRegistry
    incidence: np.ndarray
    observation: np.ndarray
    port_groups: Mapping[str, HeatPortGroup]
    allocation_families: Mapping[str, AllocationFamilySpec]
    signal_to_allocation_family: Mapping[str, str]
    resolved_adjacency: tuple[tuple[str, str], ...]
    resolved_connection_rules: tuple[tuple[str, str], ...]

    @property
    def state_index(self) -> Mapping[str, int]:
        return {node.key: i for i, node in enumerate(self.state_nodes)}

    @property
    def boundary_index(self) -> Mapping[str, int]:
        return {node.key: i for i, node in enumerate(self.boundary_nodes)}

    @property
    def port_index(self) -> Mapping[str, int]:
        return {port.key: i for i, port in enumerate(self.thermal_ports)}

    @property
    def state_dimension(self) -> int:
        return len(self.state_nodes)

    @property
    def output_dimension(self) -> int:
        return len(self.observed_nodes)

    def matrices(self, parameter_values: Mapping[str, float]) -> RCMatrices:
        values = self.parameter_registry.resolve_instance_values(parameter_values)

        C = np.empty(len(self.state_nodes), dtype=float)
        for i, node in enumerate(self.state_nodes):
            instance_id = self.state_capacitance_parameter[node.key]
            value = float(values[instance_id])
            if value <= 0.0:
                raise RCCompileError(
                    f"Capacitance {instance_id!r} must be strictly positive"
                )
            C[i] = value

        conductances = np.empty(len(self.resistance_edges), dtype=float)
        for j, edge in enumerate(self.resistance_edges):
            resistance = float(values[edge.parameter_instance_id])
            if resistance <= 0.0:
                raise RCCompileError(
                    f"Resistance {edge.parameter_instance_id!r} must be strictly positive"
                )
            conductances[j] = 1.0 / resistance

        G = np.diag(conductances)
        D = np.array(self.incidence, copy=True)
        L = D @ G @ D.T

        n_c = len(self.state_nodes)
        L_CC = L[:n_c, :n_c]
        L_CB = L[:n_c, n_c:]
        Gamma = self._build_gamma(values)

        return RCMatrices(
            C=C,
            D=D,
            G=G,
            L=L,
            L_CC=L_CC,
            L_CB=L_CB,
            Gamma=Gamma,
            H=np.array(self.observation, copy=True),
            instance_values=values,
        )

    def _build_gamma(self, values: Mapping[str, float]) -> np.ndarray:
        gamma = np.zeros((len(self.state_nodes), len(self.thermal_ports)), dtype=float)
        sidx = self.state_index

        for j, port in enumerate(self.thermal_ports):
            group = self.port_groups[port.signal]
            zone = port.zone_id
            if self.flavour.routing_kind == "all_to_air":
                gamma[sidx[StateNode(zone, "a").key], j] = 1.0
                continue

            if group is HeatPortGroup.CONVECTIVE:
                gamma[sidx[StateNode(zone, "a").key], j] = 1.0
                continue

            if self.flavour.routing_kind == "eta_r":
                eta_id = self.routing_parameter_ids[(zone, "eta_r")]
                eta = float(values[eta_id])
                if not 0.0 <= eta <= 1.0:
                    raise RCCompileError(f"{eta_id} must lie in [0,1], got {eta}")
                gamma[sidx[StateNode(zone, "a").key], j] = 1.0 - eta
                gamma[sidx[StateNode(zone, "m").key], j] = eta
                continue

            if self.flavour.routing_kind == "gamma_r_3way":
                families = ("gamma_a_r", "gamma_e_r", "gamma_m_r")
                coeffs = [
                    float(values[self.routing_parameter_ids[(zone, family)]])
                    for family in families
                ]
                if any(value < 0.0 or value > 1.0 for value in coeffs):
                    raise RCCompileError(
                        f"4R3C radiative routing coefficients for {zone!r} must lie in [0,1]"
                    )
                if abs(sum(coeffs) - 1.0) > 1e-10:
                    raise RCCompileError(
                        f"4R3C radiative routing coefficients for {zone!r} must sum to 1"
                    )
                for state_label, coefficient in zip(("a", "e", "m"), coeffs):
                    gamma[sidx[StateNode(zone, state_label).key], j] = coefficient
                continue

            raise RCCompileError(
                f"Unhandled routing kind {self.flavour.routing_kind!r}"
            )

        # Every effective thermal-power column is conservative by contract.
        if gamma.size:
            sums = gamma.sum(axis=0)
            if not np.allclose(sums, 1.0, atol=1e-10, rtol=0.0):
                raise RCCompileError(
                    f"Non-conservative heat-routing columns detected: {sums}"
                )
            if np.any(gamma < -1e-12):
                raise RCCompileError("Heat-routing matrix contains negative coefficients")
        return gamma


def _resolve_port_groups(spec: RCCompilerSpec) -> dict[str, HeatPortGroup]:
    groups = dict(DEFAULT_CORE_PORT_GROUPS)
    for signal, value in spec.port_groups.items():
        if signal == "phvac":
            raise RCCompileError("PHVAC is electrical power and cannot be a thermal port")
        if isinstance(value, HeatPortGroup):
            groups[str(signal)] = value
        else:
            try:
                groups[str(signal)] = HeatPortGroup(str(value))
            except ValueError as exc:
                raise RCCompileError(
                    f"Unknown heat-port group {value!r} for signal {signal!r}"
                ) from exc
    return groups


def _resolve_zone_ports(
    spec: RCCompilerSpec,
    groups: Mapping[str, HeatPortGroup],
) -> dict[str, tuple[str, ...]]:
    if not spec.zone_port_availability:
        return {zone: tuple(DEFAULT_CORE_PORT_GROUPS) for zone in spec.zone_ids}

    if set(spec.zone_port_availability) != set(spec.zone_ids):
        raise RCCompileError(
            "zone_port_availability, when supplied, must define every modeled zone"
        )

    out: dict[str, tuple[str, ...]] = {}
    for zone in spec.zone_ids:
        signals = tuple(str(x) for x in spec.zone_port_availability[zone])
        if len(signals) != len(set(signals)):
            raise RCCompileError(f"Duplicate thermal ports declared for zone {zone!r}")
        if "phvac" in signals:
            raise RCCompileError(
                f"Zone {zone!r} declares PHVAC as a thermal port; this is forbidden"
            )
        missing_groups = set(signals) - set(groups)
        if missing_groups:
            raise RCCompileError(
                f"Zone {zone!r} has thermal ports without routing groups: "
                f"{sorted(missing_groups)}"
            )
        out[zone] = signals
    return out


def _resolve_adjacency(spec: RCCompilerSpec) -> tuple[tuple[str, str], ...]:
    if spec.mode is SpatialMode.IND:
        # IND physical graph is uncoupled regardless of information-product lineage.
        return ()

    if spec.adjacency is None:
        return tuple(
            tuple(sorted((spec.zone_ids[i], spec.zone_ids[i + 1])))
            for i in range(len(spec.zone_ids) - 1)
        )

    valid_zones = set(spec.zone_ids)
    resolved: set[tuple[str, str]] = set()
    for declaration in spec.adjacency:
        pair = declaration.canonical()
        if not set(pair).issubset(valid_zones):
            raise RCCompileError(
                f"Adjacency {pair} references a zone outside modeled zone_ids"
            )
        resolved.add(pair)
    return tuple(sorted(resolved))


def _resolve_connection_rules(
    spec: RCCompilerSpec,
    flavour: RCFlavor,
) -> tuple[tuple[str, str], ...]:
    declarations: Sequence[ConnectionRule]
    if spec.connection_rules:
        declarations = spec.connection_rules
    else:
        declarations = (ConnectionRule("a", "a"),)

    available = set(flavour.state_labels)
    resolved: set[tuple[str, str]] = set()
    for declaration in declarations:
        pair = declaration.canonical()
        if not set(pair).issubset(available):
            raise RCCompileError(
                f"Connection rule {pair} references a state not present in "
                f"{flavour.name}: {flavour.state_labels}"
            )
        resolved.add(pair)
    return tuple(sorted(resolved))


def _add_edge(
    edges: dict[tuple[str, str], ResistanceEdge],
    edge: ResistanceEdge,
) -> None:
    key = edge.endpoint_keys()
    previous = edges.get(key)
    if previous is None:
        edges[key] = edge
        return
    if (
        previous.family == edge.family
        and previous.parameter_instance_id == edge.parameter_instance_id
        and previous.kind == edge.kind
    ):
        return
    raise RCCompileError(
        f"Conflicting duplicate physical edge {key}: "
        f"{previous.family!r} versus {edge.family!r}"
    )


def _allocation_contracts(
    spec: RCCompilerSpec,
    zone_ports: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, AllocationFamilySpec], dict[str, str]]:
    if spec.mode is not SpatialMode.DEP2:
        if spec.dep2_allocations:
            raise RCCompileError("DEP2 allocation families may only be supplied in DEP2 mode")
        return {}, {}

    families: dict[str, AllocationFamilySpec] = {}
    signal_owner: dict[str, str] = {}
    for family in spec.dep2_allocations:
        if family.name in families:
            raise RCCompileError(f"Duplicate allocation family name: {family.name!r}")
        validate_allocation_spec(family, spec.zone_ids)
        if "qac" in family.signals:
            raise RCCompileError("QAC must remain local and cannot be DEP2 allocated")
        for signal in family.signals:
            if signal in signal_owner:
                raise RCCompileError(
                    f"DEP2 signal {signal!r} belongs to multiple allocation families"
                )
            signal_owner[signal] = family.name

        participants = (
            set(family.participating_zone_ids)
            if family.participating_zone_ids
            else set(spec.zone_ids)
        )
        for signal in family.signals:
            actual = {zone for zone in spec.zone_ids if signal in zone_ports[zone]}
            if actual != participants:
                raise RCCompileError(
                    f"Signals sharing DEP2 family {family.name!r} must have exactly "
                    f"the participating-zone set. signal={signal!r}, "
                    f"expected={sorted(participants)}, found={sorted(actual)}"
                )
        families[family.name] = family

    required_non_hvac = {
        signal
        for zone in spec.zone_ids
        for signal in zone_ports[zone]
        if signal != "qac"
    }
    missing = required_non_hvac - set(signal_owner)
    extra = set(signal_owner) - required_non_hvac
    if missing:
        raise RCCompileError(
            "DEP2 requires explicit allocation family coverage for every applicable "
            f"non-HVAC thermal signal: missing={sorted(missing)}"
        )
    if extra:
        raise RCCompileError(
            f"DEP2 allocation families reference unused thermal signals: {sorted(extra)}"
        )
    return families, signal_owner


def compile_rc_model(
    spec: RCCompilerSpec,
    *,
    parameter_configs: Mapping[str, ParameterConfig] | None = None,
) -> CompiledRCModel:
    """Compile one frozen RC specification into a deterministic graph contract."""

    flavour = get_flavour(spec.flavour)
    groups = _resolve_port_groups(spec)
    zone_ports = _resolve_zone_ports(spec, groups)
    adjacency = _resolve_adjacency(spec)
    connection_rules = _resolve_connection_rules(spec, flavour)
    allocations, signal_owner = _allocation_contracts(spec, zone_ports)

    state_nodes = tuple(
        StateNode(zone, label)
        for zone in spec.zone_ids
        for label in flavour.state_labels
    )
    state_keys = {node.key for node in state_nodes}
    boundary_labels = tuple(
        dict.fromkeys(edge.boundary_label for edge in flavour.boundary_edges)
    )
    boundary_nodes = tuple(BoundaryNode(label) for label in boundary_labels)

    parameter_instances: list[ParameterInstance] = []
    cap_parameter: dict[str, str] = {}

    for zone in spec.zone_ids:
        for state_label in flavour.state_labels:
            family = flavour.capacitance_families[state_label]
            instance = make_capacitance_instance(zone, family)
            parameter_instances.append(instance)
            cap_parameter[StateNode(zone, state_label).key] = instance.instance_id

    edge_lookup: dict[tuple[str, str], ResistanceEdge] = {}

    for zone in spec.zone_ids:
        for template in flavour.state_edges:
            a = StateNode(zone, template.node_a)
            b = StateNode(zone, template.node_b)
            endpoints = tuple(sorted((a.key, b.key)))
            discriminator = f"elem|{endpoints[0]}--{endpoints[1]}"
            instance = make_resistance_instance(
                family=template.resistance_family,
                zone_scope=(zone,),
                discriminator=discriminator,
            )
            parameter_instances.append(instance)
            _add_edge(
                edge_lookup,
                ResistanceEdge(
                    edge_id=f"edge|{discriminator}",
                    node_a=a,
                    node_b=b,
                    family=template.resistance_family,
                    parameter_instance_id=instance.instance_id,
                    kind="elemental_state",
                ),
            )

        for template in flavour.boundary_edges:
            a = StateNode(zone, template.state_label)
            b = BoundaryNode(template.boundary_label)
            discriminator = f"boundary|{a.key}--{b.key}"
            instance = make_resistance_instance(
                family=template.resistance_family,
                zone_scope=(zone,),
                discriminator=discriminator,
            )
            parameter_instances.append(instance)
            _add_edge(
                edge_lookup,
                ResistanceEdge(
                    edge_id=f"edge|{discriminator}",
                    node_a=a,
                    node_b=b,
                    family=template.resistance_family,
                    parameter_instance_id=instance.instance_id,
                    kind="elemental_boundary",
                ),
            )

    if spec.mode in {SpatialMode.DEP1, SpatialMode.DEP2}:
        for zone_a, zone_b in adjacency:
            for state_a, state_b in connection_rules:
                pairs: tuple[tuple[str, str], ...]
                if state_a == state_b:
                    pairs = ((state_a, state_b),)
                else:
                    pairs = ((state_a, state_b), (state_b, state_a))

                for left_state, right_state in pairs:
                    a = StateNode(zone_a, left_state)
                    b = StateNode(zone_b, right_state)
                    if a.key not in state_keys or b.key not in state_keys:
                        raise RCCompileError("Generated inter-zone state edge is invalid")
                    endpoints = tuple(sorted((a.key, b.key)))
                    family_states = tuple(sorted((state_a, state_b)))
                    family = f"R_inter_{family_states[0]}_{family_states[1]}"
                    discriminator = f"inter|{endpoints[0]}--{endpoints[1]}"
                    instance = make_resistance_instance(
                        family=family,
                        zone_scope=(zone_a, zone_b),
                        discriminator=discriminator,
                    )
                    parameter_instances.append(instance)
                    _add_edge(
                        edge_lookup,
                        ResistanceEdge(
                            edge_id=f"edge|{discriminator}",
                            node_a=a,
                            node_b=b,
                            family=family,
                            parameter_instance_id=instance.instance_id,
                            kind="inter_zone",
                        ),
                    )

    thermal_ports = tuple(
        ThermalPort(zone, signal)
        for zone in spec.zone_ids
        for signal in zone_ports[zone]
    )

    routing_parameter_ids: dict[tuple[str, str], str] = {}
    for zone in spec.zone_ids:
        has_radiative = any(
            groups[signal] is HeatPortGroup.RADIATIVE for signal in zone_ports[zone]
        )
        if not has_radiative:
            continue
        if flavour.routing_kind == "eta_r":
            instance = make_routing_instance(zone, "eta_r")
            parameter_instances.append(instance)
            routing_parameter_ids[(zone, "eta_r")] = instance.instance_id
        elif flavour.routing_kind == "gamma_r_3way":
            for family in ("gamma_a_r", "gamma_e_r", "gamma_m_r"):
                instance = make_routing_instance(zone, family)
                parameter_instances.append(instance)
                routing_parameter_ids[(zone, family)] = instance.instance_id

    registry = build_parameter_registry(
        parameter_instances,
        spec.parameter_sharing,
        parameter_configs,
    )

    resistance_edges = tuple(
        sorted(edge_lookup.values(), key=lambda edge: edge.endpoint_keys())
    )

    all_nodes: tuple[StateNode | BoundaryNode, ...] = state_nodes + boundary_nodes
    node_index = {node.key: i for i, node in enumerate(all_nodes)}
    incidence = np.zeros((len(all_nodes), len(resistance_edges)), dtype=float)
    for j, edge in enumerate(resistance_edges):
        # Deterministic algebraic orientation by lexical endpoint key.
        if edge.node_a.key <= edge.node_b.key:
            plus, minus = edge.node_a, edge.node_b
        else:
            plus, minus = edge.node_b, edge.node_a
        incidence[node_index[plus.key], j] = 1.0
        incidence[node_index[minus.key], j] = -1.0

    observed_nodes = tuple(
        StateNode(zone, label)
        for zone in spec.zone_ids
        for label in flavour.observed_state_labels
    )
    state_index = {node.key: i for i, node in enumerate(state_nodes)}
    observation = np.zeros((len(observed_nodes), len(state_nodes)), dtype=float)
    for row, node in enumerate(observed_nodes):
        observation[row, state_index[node.key]] = 1.0

    return CompiledRCModel(
        spec=spec,
        flavour=flavour,
        state_nodes=state_nodes,
        boundary_nodes=boundary_nodes,
        resistance_edges=resistance_edges,
        thermal_ports=thermal_ports,
        observed_nodes=observed_nodes,
        state_capacitance_parameter=cap_parameter,
        routing_parameter_ids=routing_parameter_ids,
        parameter_registry=registry,
        incidence=incidence,
        observation=observation,
        port_groups={signal: groups[signal] for signal in groups},
        allocation_families=allocations,
        signal_to_allocation_family=signal_owner,
        resolved_adjacency=adjacency,
        resolved_connection_rules=connection_rules,
    )
