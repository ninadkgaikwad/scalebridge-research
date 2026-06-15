import importlib
import site
import sys

print("=" * 80)
print("PIL / TorchVision / Matplotlib DLL diagnostic")
print("=" * 80)
print("Python:", sys.version)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
print("USER_SITE:", site.getusersitepackages())
print()

def check(name, code):
    print("-" * 80)
    print(f"TEST: {name}")
    try:
        exec(code, {})
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")

check("Pillow direct", """
from PIL import Image
import PIL._imaging
print("PIL version:", Image.__version__)
print("PIL path:", Image.__file__)
print("_imaging path:", PIL._imaging.__file__)
""")

check("matplotlib direct", """
import matplotlib
print("matplotlib version:", matplotlib.__version__)
print("matplotlib path:", matplotlib.__file__)
""")

check("torch direct", """
import torch
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
""")

check("torch then Pillow", """
import torch
from PIL import Image
import PIL._imaging
print("torch then Pillow OK")
""")

check("Pillow then torchvision", """
from PIL import Image
import PIL._imaging
import torchvision
print("torchvision version:", torchvision.__version__)
print("torchvision path:", torchvision.__file__)
""")

check("torchvision direct", """
import torchvision
print("torchvision version:", torchvision.__version__)
print("torchvision path:", torchvision.__file__)
""")

check("stable_baselines3", """
import stable_baselines3
print("stable_baselines3 OK")
""")

check("pytorch_lightning", """
import pytorch_lightning
print("pytorch_lightning OK")
""")

check("neuromancer", """
import neuromancer
print("neuromancer OK")
""")

print("=" * 80)
print("Diagnostic complete")
print("=" * 80)
