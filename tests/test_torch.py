import torch

print("Disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA (torch):", torch.version.cuda)

# Test de uso real
x = torch.rand(3, 3).to("cuda")
print("Tensor en GPU:", x)