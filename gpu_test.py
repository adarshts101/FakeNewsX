import torch

print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
