import torch

print("torch version:", torch.__version__)
print("is cuda available:", torch.cuda.is_available())
print("available gpu num:", torch.cuda.device_count())
print("device name:", torch.cuda.get_device_name())
print("arch list:", torch.cuda.get_arch_list())
print("device capability:", torch.cuda.get_device_capability())