import json
import torch
import random
import argparse
import numpy as np


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
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


def get_result_path(config):
    return (
        f"output/"
        f"_in-{config['model']['input_size']}"
        f"_hd-{'-'.join(str(size) for size in config['model']['hidden_sizes'])}"
        f"_out-{config['model']['output_size']}"
        f"_dr-{config['model']['dropout_rate']}"
        f"_strategy-{config['prune']['strategy']}"
        f"_type-{config['prune']['params']['type']}"
        f"_p-{config['prune']['params']['rate']}"
        f".pkl"
    )


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
