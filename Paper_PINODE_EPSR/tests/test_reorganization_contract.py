from __future__ import annotations

from pathlib import Path
import importlib


def test_canonical_src_layout_exists():
    root = Path(__file__).parents[1]
    expected = [
        "src/pinode_epsr/core/common.py",
        "src/pinode_epsr/core/config.py",
        "src/pinode_epsr/physics/rc.py",
        "src/pinode_epsr/physics/energy_projection.py",
        "src/pinode_epsr/data/phase_d.py",
        "src/pinode_epsr/data/phase_c.py",
        "src/pinode_epsr/backends/neuromancer.py",
        "src/pinode_epsr/methods/inverse_pinn.py",
        "src/pinode_epsr/methods/neural_ode.py",
        "src/pinode_epsr/methods/base_pinode.py",
        "src/pinode_epsr/methods/ebp_pinode.py",
        "src/pinode_epsr/training/trainer.py",
        "src/pinode_epsr/evaluation/runtime.py",
        "src/pinode_epsr/evaluation/thermostat.py",
    ]
    for rel in expected:
        assert (root / rel).is_file(), rel


def test_historical_imports_are_compatibility_shims_to_canonical_classes():
    old = importlib.import_module("Paper_PINODE_EPSR.ebp_pinode")
    new = importlib.import_module("pinode_epsr.methods.ebp_pinode")
    assert old.EBPPINODEModel is new.EBPPINODEModel
    assert old.EBPPINODEConfig is new.EBPPINODEConfig


def test_all_four_method_shims_resolve_to_canonical_implementations():
    pairs = [
        ("Paper_PINODE_EPSR.inverse_pinn", "pinode_epsr.methods.inverse_pinn", "InversePINNRC"),
        ("Paper_PINODE_EPSR.neural_ode", "pinode_epsr.methods.neural_ode", "NeuralODEModel"),
        ("Paper_PINODE_EPSR.base_pinode", "pinode_epsr.methods.base_pinode", "BasePINODEModel"),
        ("Paper_PINODE_EPSR.ebp_pinode", "pinode_epsr.methods.ebp_pinode", "EBPPINODEModel"),
    ]
    for old_name, new_name, symbol in pairs:
        old = importlib.import_module(old_name)
        new = importlib.import_module(new_name)
        assert getattr(old, symbol) is getattr(new, symbol)


def test_energy_projection_is_first_class_physics_module():
    old = importlib.import_module("Paper_PINODE_EPSR.ebp_pinode")
    physics = importlib.import_module("pinode_epsr.physics.energy_projection")
    assert old.weighted_energy_projection is physics.weighted_energy_projection


def test_patch_notes_are_archived_outside_public_root():
    root = Path(__file__).parents[1]
    notes = root / "development_history" / "patch_notes"
    assert (notes / "PATCH04_NOTES.txt").is_file()
    assert not (root / "PATCH04_NOTES.txt").exists()


def test_approved_validation_records_are_grouped_under_validation():
    root = Path(__file__).parents[1]
    assert (root / "validation" / "patch04_validation_real.json").is_file()
    assert not (root / "results" / "patch04_validation_real.json").exists()


def test_standalone_pyproject_uses_src_layout():
    root = Path(__file__).parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "pinode-epsr"' in text
    assert 'package-dir = {"" = "src"}' in text


def test_public_readme_identifies_canonical_package_and_scalebridge_embedding():
    root = Path(__file__).parents[1]
    text = (root / "README.md").read_text(encoding="utf-8")
    assert "src/pinode_epsr" in text
    assert "scalebridge-research" in text
    assert "compatibility" in text.lower()
