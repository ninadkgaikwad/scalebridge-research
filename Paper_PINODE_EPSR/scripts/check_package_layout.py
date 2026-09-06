from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[2]
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

import Paper_PINODE_EPSR as compat
import pinode_epsr as canonical

assert compat.EBPPINODEModel is canonical.EBPPINODEModel
assert compat.BasePINODEModel is canonical.BasePINODEModel
print("PINODE EPSR package-layout parity: PASSED")
