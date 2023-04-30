import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_model_path(seed, dropout_rate, use_1d):
    return (
        f"models/"
        f"lenet-300-100"
        f"_seed-{seed}"
        f"_dropout_rate-{dropout_rate}"
        f"_use_1d-{use_1d}"
        f".pth"
    )