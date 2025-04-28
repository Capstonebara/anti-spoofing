import torch
print("CUDA version reported by torch:", torch.version.cuda)
print("CUDA is available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
