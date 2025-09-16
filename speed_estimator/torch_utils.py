import torch
import subprocess

# -----------------------------------------------------------------------------
# torch_utils.py
# Utility functions for GPU selection and monitoring.
# - get_gpu_memory_map(): query nvidia-smi for free/total memory.
# - select_gpu_with_most_free_memory(): auto-select CUDA device with max free MB.
# NOTE:
#   Requires 'nvidia-smi' in PATH (NVIDIA driver + toolkit installed).
#   Intended to help multi-GPU training scripts choose the best device.
# -----------------------------------------------------------------------------


def get_gpu_memory_map():
    """Get the current GPU usage using nvidia-smi.

    Returns
    -------
    memory_map: dict
        Keys are device ids as integers.
        Values are memory free on that device in MB as integers.
    
    Query nvidia-smi to get free/total memory per GPU.
    Returns:
        dict: { device_id (int) : free_memory_MB (int) }
    Notes:
        - Uses subprocess to call 'nvidia-smi'.
        - Parses output as CSV (no units, no headers).
        - Fails if nvidia-smi is missing or driver not installed.
    """
    # Run nvidia-smi command to get memory information
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'], encoding='utf-8'
    )

    # Extract the memory information
    gpu_memory = [tuple(map(int, x.split(','))) for x in result.strip().split('\n')]

    # Create a memory map as a dictionary
    memory_map = {i: free for i, (free, total) in enumerate(gpu_memory)}
    return memory_map

def select_gpu_with_most_free_memory():
    """
    Select the GPU with the most available memory.
    
    Select CUDA device with the most free memory.
    Returns:
        int or None:
          - index of selected GPU (if CUDA available),
          - None if no GPU is visible.
    Side effects:
        - Sets torch.cuda device via torch.cuda.set_device(best_gpu).
        - Prints which GPU was selected and free MB available.
    """
    if torch.cuda.is_available():
        memory_map = get_gpu_memory_map()
        best_gpu = max(memory_map, key=memory_map.get)
        torch.cuda.set_device(best_gpu)
        print(f"Selected GPU {best_gpu} with {memory_map[best_gpu]} MB free memory.")
        return best_gpu
    else:
        print("No GPU available, using CPU.")
        return None


# GOTCHA:
# - Works only if NVIDIA GPUs + drivers + nvidia-smi are available.
# - Free memory reported is "at query time"; may change immediately after.
# - In multi-process training, race conditions possible (two processes pick same GPU).
# - If running on cluster with job scheduler, prefer using CUDA_VISIBLE_DEVICES.
