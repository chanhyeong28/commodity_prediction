import os
import random
from itertools import islice

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_gpu_stats():
    # Print GPU stats
    device = torch.cuda.current_device()
    memory_stats = torch.cuda.memory_stats(device=device)
    t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated_memory_gb = memory_stats["allocated_bytes.all.current"] / (1024**3)
    print(f"Total Memory: {t:.2f} GB")
    print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")