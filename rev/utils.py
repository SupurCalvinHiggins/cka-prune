import json
import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def get_model_path(model, seed):
    return (
        f"models/"
        f"mlp"
        f"_in-{model.input_size}"
        f"_hd-{'-'.join(model.hidden_sizes)}"
        f"_out-{model.output_size}"
        f"_dr-{model.dropout_rate}"
        f"_seed-{seed}"
        f".pth"
    )