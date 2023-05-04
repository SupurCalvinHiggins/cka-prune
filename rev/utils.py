import json
import torch
import random
import argparse
import numpy as np


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--output_name', default="", type=str)
    return parser


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def get_model_path(model_config, seed):
    return (
        f"models/"
        f"mlp"
        f"_in-{model_config['input_size']}"
        f"_hd-{'-'.join(str(size) for size in model_config['hidden_sizes'])}"
        f"_out-{model_config['output_size']}"
        f"_dr-{model_config['dropout_rate']}"
        f"_seed-{seed}"
        f".pth"
    )


def get_result_path(id):
    return (
        f"output/"
        f"id-{id}"
        f".pkl"
    )


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
