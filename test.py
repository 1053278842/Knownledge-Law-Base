import torch

print(torch.__version__)
print(torch.cuda.is_available())   # 应该是 False
print(torch.version.cuda)          # 应该是 None