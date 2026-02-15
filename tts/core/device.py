import torch
from runpod.serverless.utils import rp_cuda

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if rp_cuda.is_available() else "cpu"
