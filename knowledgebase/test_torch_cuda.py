import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.randn(500, 500, device="cuda")
    y = x @ x
    print("gpu matmul OK:", y.shape)
else:
    raise RuntimeError("CUDA is not available to PyTorch")
