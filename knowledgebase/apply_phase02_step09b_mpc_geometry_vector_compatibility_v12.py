from pathlib import Path
import datetime
import shutil

repo = Path.cwd()
helper = repo / "code/Experiments/Exp_Modules/Exp_MPC_RL_Helpers.py"
test = repo / "tests/integration/experiment_compatibility/test_step9b_mpc_geometry_vector_compatibility.py"

if not helper.is_file():
    raise SystemExit(f"Missing helper source: {helper}")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_root = repo / "development/phase_02/step09b_backups" / stamp
backup = backup_root / helper.relative_to(repo)
backup.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(helper, backup)

text = helper.read_text(encoding="utf-8-sig")

func_anchor = "def compute_singlehouse_RC_data_from_ctx(ctx):"
compat_helper = '''def _as_legacy_vector(values, name):
    """Normalize a legacy/Python parameter vector to canonical 1-D form.

    Accepted:
      - Python-native ``(n,)``
      - MATLAB-style row ``(1,n)``
      - MATLAB-style column ``(n,1)``

    This is intended for scalar-per-surface/element vectors such as window
    tilt and azimuth. It does not reinterpret multi-row/multi-column matrices.
    """
    array = np.asarray(values)

    if array.ndim == 1:
        return array

    if array.ndim == 2 and array.shape[0] == 1:
        return array[0, :]

    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]

    raise ValueError(
        f"{name} must have shape (n,), (1,n), or (n,1); got {array.shape}."
    )


'''

if "_as_legacy_vector" not in text:
    if func_anchor not in text:
        raise SystemExit("Could not locate compute_singlehouse_RC_data_from_ctx().")
    text = text.replace(func_anchor, compat_helper + func_anchor, 1)

old = "_ViewFactor(beta, phi, Tilt_w[0,jj], Azi_w[0,jj])"
new = '_ViewFactor(beta, phi, _as_legacy_vector(Tilt_w, "Tilt_w")[jj], _as_legacy_vector(Azi_w, "Azi_w")[jj])'

count = text.count(old)
if count == 0:
    raise SystemExit(
        "Could not locate the expected Tilt_w/Azi_w legacy indexing expression. "
        "No source changes were made."
    )

text = text.replace(old, new)
helper.write_text(text, encoding="utf-8")

test.parent.mkdir(parents=True, exist_ok=True)
test.write_text(
'''import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "code/Experiments/Exp_Modules/Exp_MPC_RL_Helpers.py"


def _load_helpers():
    fake_config = types.ModuleType("Exp_Config_Module")
    sys.modules.setdefault("Exp_Config_Module", fake_config)

    fake_smartcom = types.ModuleType("SmartComSim")
    fake_smartcom.SmartCommunity_Simulator = object
    sys.modules.setdefault("SmartComSim", fake_smartcom)

    spec = importlib.util.spec_from_file_location(
        "step9b_geometry_helpers", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helpers():
    return _load_helpers()


@pytest.mark.parametrize(
    "array",
    [
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([[10.0, 20.0, 30.0, 40.0]]),
        np.array([[10.0], [20.0], [30.0], [40.0]]),
    ],
)
def test_geometry_vector_accepts_python_and_matlab_shapes(helpers, array):
    result = helpers._as_legacy_vector(array, "geometry")
    assert result.shape == (4,)
    np.testing.assert_allclose(result, [10.0, 20.0, 30.0, 40.0])


def test_geometry_vector_rejects_true_matrix(helpers):
    with pytest.raises(ValueError):
        helpers._as_legacy_vector(np.zeros((2, 2)), "geometry")


def test_viewfactor_call_no_longer_assumes_matlab_row_vector():
    text = MODULE_PATH.read_text(encoding="utf-8-sig")
    assert 'Tilt_w[0,jj]' not in text
    assert 'Azi_w[0,jj]' not in text
    assert '_as_legacy_vector(Tilt_w, "Tilt_w")[jj]' in text
    assert '_as_legacy_vector(Azi_w, "Azi_w")[jj]' in text
''',
encoding="utf-8",
)

manifest = repo / (
    "outputs/phase_02_python_backend_ev/manifests/"
    "step_09b_mpc_geometry_vector_compatibility_v12_manifest.txt"
)
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(
    "SMARTCOMMUNITYSIM_2 PHASE 2 STEP 9B MPC GEOMETRY VECTOR COMPATIBILITY V12\n"
    "=======================================================================\n\n"
    f"Applied: {datetime.datetime.now().isoformat()}\n"
    f"Backup: {backup_root}\n\n"
    "Scope:\n"
    "- Shared MPC helper only.\n"
    "- No simulator changes.\n"
    "- No MPC formulation/equation changes.\n"
    "- No RL changes.\n"
    "- No controller contract changes.\n\n"
    "Compatibility:\n"
    "- Tilt_w/Azi_w accept Python (n,), MATLAB row (1,n), or MATLAB column (n,1).\n"
    "- Same scalar element jj is passed to _ViewFactor.\n",
    encoding="utf-8",
)

print("Step 9B MPC geometry-vector compatibility V12 applied.")
print(f"Replaced {count} legacy Tilt_w/Azi_w indexing occurrence(s).")
print("Backup:", backup_root)
print("Manifest:", manifest)
