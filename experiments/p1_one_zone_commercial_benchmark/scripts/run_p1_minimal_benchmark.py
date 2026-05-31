"""Minimal P1 benchmark runner placeholder.

Target first milestone:
    one building-climate case
    -> clean one-zone dataset
    -> PyTorch MLP baseline
    -> metrics
    -> MLflow run
"""

from scalebridge.models.black_box.mlp.model import MLPRegressor


def main() -> None:
    print("P1 minimal benchmark placeholder. Next: connect dataset loader and trainer.")
    _ = MLPRegressor(input_dim=10, output_dim=1, hidden_dims=[64, 64])


if __name__ == "__main__":
    main()
