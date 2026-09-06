# Reproducibility

Validated framework: Neuromancer 1.5.6 with PyTorch autograd.
No custom RK4 and no direct TorchDiffEq integration are used in the paper methods. EBP uses differentiable `torch.linalg.solve`.

The compatibility/regression test suite must pass after structural reorganization before scientific development continues.
