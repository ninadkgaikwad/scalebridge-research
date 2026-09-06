from __future__ import annotations

"""Thin, auditable NeuroMANCER adapter for the EPSR paper implementation.

Framework contract for Patch 02:
  * NeuroMANCER blocks.MLP provides trainable neural function approximators.
  * NeuroMANCER Node/System provides recursive computational graphs.
  * NeuroMANCER dynamics.integrators.RK4 provides ALL numerical ODE steps.
  * PyTorch provides tensors, autograd, parameter transforms and optimizers.
  * torchdiffeq is NOT called directly by paper code.

No Runge--Kutta formula is implemented in this repository.  The only RK4
implementation used by paper code is NeuroMANCER's published RK4 class.
"""

from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from neuromancer.constraint import variable as neuromancer_variable
    from neuromancer.dataset import DictDataset as NeuromancerDictDataset
    from neuromancer.dynamics.integrators import RK4 as NeuromancerRK4
    from neuromancer.loss import PenaltyLoss as NeuromancerPenaltyLoss
    from neuromancer.modules.blocks import MLP as NeuromancerMLP
    from neuromancer.problem import Problem as NeuromancerProblem
    from neuromancer.system import Node as NeuromancerNode
    from neuromancer.system import System as NeuromancerSystem
except Exception as exc:  # pragma: no cover - exercised on target ScaleBridge env
    raise RuntimeError(
        "PINODE/EPSR Patch 02 requires NeuroMANCER. Activate the qualified "
        "ScaleBridge environment containing neuromancer before importing the "
        "paper SciML methods."
    ) from exc


@dataclass(frozen=True)
class NeuromancerRuntime:
    version: str
    rk4_class: str
    node_class: str
    system_class: str
    mlp_class: str
    problem_class: str
    penalty_loss_class: str


def runtime_info() -> NeuromancerRuntime:
    try:
        version = metadata.version("neuromancer")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return NeuromancerRuntime(
        version=version,
        rk4_class=f"{NeuromancerRK4.__module__}.{NeuromancerRK4.__name__}",
        node_class=f"{NeuromancerNode.__module__}.{NeuromancerNode.__name__}",
        system_class=f"{NeuromancerSystem.__module__}.{NeuromancerSystem.__name__}",
        mlp_class=f"{NeuromancerMLP.__module__}.{NeuromancerMLP.__name__}",
        problem_class=f"{NeuromancerProblem.__module__}.{NeuromancerProblem.__name__}",
        penalty_loss_class=f"{NeuromancerPenaltyLoss.__module__}.{NeuromancerPenaltyLoss.__name__}",
    )


def activation_class(name: str) -> type[nn.Module]:
    table: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "softplus": nn.Softplus,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported activation {name!r}; choose {sorted(table)}")
    return table[key]


def build_mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_layers: int,
    hidden_width: int,
    activation: str = "tanh",
    dtype: torch.dtype = torch.float64,
) -> nn.Module:
    """Create a NeuroMANCER MLP with paper-controlled architecture size."""

    if input_dim < 1 or output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive")
    if hidden_layers < 1 or hidden_width < 1:
        raise ValueError("hidden_layers and hidden_width must be positive")
    net = NeuromancerMLP(
        insize=input_dim,
        outsize=output_dim,
        nonlin=activation_class(activation),
        hsizes=[hidden_width] * hidden_layers,
    )
    return net.to(dtype=dtype)


class _CallableODEBlock(nn.Module):
    """Adapter satisfying NeuroMANCER Integrator block metadata requirements."""

    def __init__(self, fn: Callable[..., torch.Tensor], state_dim: int, extra_dim: int = 0) -> None:
        super().__init__()
        self.fn = fn
        self.in_features = int(state_dim + extra_dim)
        self.out_features = int(state_dim)

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        return self.fn(x, *args)


def rk4_interval(
    rhs: Callable[..., torch.Tensor],
    x: torch.Tensor,
    *args: torch.Tensor,
    state_dim: int,
    n_substeps: int,
    interval_length: float,
    extra_dim: int = 0,
) -> torch.Tensor:
    """Integrate one sampled interval exclusively with NeuroMANCER RK4.

    ``n_substeps`` is the paper's N_s.  We instantiate NeuroMANCER's RK4 with
    h=interval_length/N_s and call that published integrator N_s times under
    zero-order-held forcing.  This wrapper contains no numerical RK formula.
    """

    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    if interval_length <= 0:
        raise ValueError("interval_length must be positive")
    block = _CallableODEBlock(rhs, state_dim=state_dim, extra_dim=extra_dim)
    integrator = NeuromancerRK4(block, h=float(interval_length) / float(n_substeps))
    out = x
    for _ in range(int(n_substeps)):
        out = integrator(out, *args)
    return out


def node(callable_module: nn.Module | Callable, input_keys: list[str], output_keys: list[str], *, name: str):
    return NeuromancerNode(callable_module, input_keys, output_keys, name=name)


def system(nodes: list[nn.Module], *, nstep_key: str, name: str, nsteps: int | None = None):
    return NeuromancerSystem(nodes, name=name, nstep_key=nstep_key, nsteps=nsteps)


class _ScalarLossClosureModule(nn.Module):
    """Expose a paper scalar loss closure as a NeuroMANCER Node output.

    The model is registered as a submodule so NeuroMANCER Problem owns the
    trainable parameters.  The anchor is only a graph input required by Node;
    it is not a mathematical input to the paper objective.
    """

    def __init__(self, model: nn.Module, loss_closure: Callable[[], torch.Tensor]) -> None:
        super().__init__()
        self.model = model
        self.loss_closure = loss_closure

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        loss = self.loss_closure()
        if loss.ndim != 0:
            loss = loss.mean()
        return loss.reshape(1, 1).expand(anchor.shape[0], 1)


def named_dataloader(
    datadict: Mapping[str, torch.Tensor],
    *,
    name: str,
    batch_size: int | None = None,
    shuffle: bool = False,
) -> DataLoader:
    """Build a NeuroMANCER-named DataLoader using the official DictDataset contract.

    NeuroMANCER ``Problem.forward`` requires ``data["name"]`` and prefixes
    every output with that dataset name.  Official NeuroMANCER examples create
    ``DictDataset(..., name="train")`` and pass ``dataset.collate_fn`` to a
    PyTorch ``DataLoader``; the collate function injects the required ``name``
    field into each batch.

    This helper centralizes that contract so paper training never hand-inserts
    metadata into raw dictionaries.
    """

    dataset_name = str(name).strip()
    if not dataset_name:
        raise ValueError("NeuroMANCER dataset name must be a non-empty string")
    tensors = {str(key): value for key, value in datadict.items()}
    if not tensors:
        raise ValueError("datadict must contain at least one tensor")
    if not all(torch.is_tensor(value) for value in tensors.values()):
        raise TypeError("all named_dataloader values must be torch.Tensor instances")

    dataset = NeuromancerDictDataset(tensors, name=dataset_name)
    resolved_batch_size = len(dataset) if batch_size is None else int(batch_size)
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=bool(shuffle),
    )


def scalar_objective_problem(
    model: nn.Module,
    loss_closure: Callable[[], torch.Tensor],
    *,
    dataset_name: str = "train",
) -> tuple[nn.Module, DataLoader]:
    """Build the NeuroMANCER Problem and an officially named one-batch loader.

    NeuroMANCER owns the objective graph.  A standard PyTorch optimizer is
    intentionally used by ``training.optimize_steps`` to update the Problem
    parameters, preserving the agreed framework division.  The returned loader
    follows NeuroMANCER's documented ``DictDataset`` + ``collate_fn`` pattern,
    which injects the dataset ``name`` expected by ``Problem.forward``.
    """

    wrapper = _ScalarLossClosureModule(model, loss_closure)
    loss_node = NeuromancerNode(
        wrapper, ["paper_anchor"], ["paper_loss_raw"], name="PaperLossNode"
    )
    paper_loss_raw = neuromancer_variable("paper_loss_raw")
    objective = paper_loss_raw.minimize(weight=1.0)
    penalty_loss = NeuromancerPenaltyLoss(objectives=[objective], constraints=[])
    problem = NeuromancerProblem(nodes=[loss_node], loss=penalty_loss)

    try:
        ref = next(model.parameters())
        anchor = torch.zeros((1, 1), dtype=ref.dtype, device=ref.device)
    except StopIteration:
        anchor = torch.zeros((1, 1), dtype=torch.float64)

    loader = named_dataloader(
        {"paper_anchor": anchor},
        name=dataset_name,
        batch_size=1,
        shuffle=False,
    )
    return problem, loader
