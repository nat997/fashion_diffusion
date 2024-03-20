import torch
gpu_id = torch.cuda.current_device()
print("GPU ID:", gpu_id)
# Print torch module path
print(torch.__path__)

# Check if CUDA is available
print(torch.cuda.is_available())

# Print the list of supported GPU architectures
print(torch.cuda.get_arch_list())

# Print the number of available GPU devices
print(torch.cuda.device_count())
