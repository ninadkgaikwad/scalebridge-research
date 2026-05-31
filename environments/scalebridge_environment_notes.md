# ScaleBridge Environment Notes

## Main files

- `scalebridge.yaml`: GPU-preferred unified conda environment.
- `scalebridge_cpu.yaml`: CPU-only fallback.
- `check_scalebridge_environment.py`: smoke test after installation.

## Recommended install

```powershell
conda env create -f scalebridge.yaml
conda activate scalebridge-dev
python check_scalebridge_environment.py
```

## CPU fallback

```powershell
conda env create -f scalebridge_cpu.yaml
conda activate scalebridge-dev-cpu
python check_scalebridge_environment.py
```

## Important rule

Do not include TensorFlow in the core ScaleBridge environment.
Use legacy TensorFlow scripts only as reference logic.
For PyTorch-compatible logging, use TensorBoard / tensorboardX instead.
