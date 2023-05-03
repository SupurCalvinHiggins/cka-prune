import os
from mlp import MLP
from torch import nn
from loaders import get_loaders
from engine import train_model, evaluate_model
from utils import *


def main(config):
    print(config)
    print()

    for seed in config["seeds"]:

        # Set seed.
        print("*** Pruning ***")
        print(f"seed = {seed}")
        set_seed(seed)

        # Get model path.
        model_path = get_model_path(config["model"], seed)
        print(f"model_path = {model_path}")

        # Skip models that don't exist.
        if not os.path.exists(model_path):
            print("model does not exist")
            print()
            continue
            
        # Set up model.
        model = MLP(
            input_size=config["model"]["input_size"],
            hidden_sizes=config["model"]["hidden_sizes"],
            output_size=config["model"]["output_size"],
            dropout_rate=config["model"]["dropout_rate"],
        )
        model.load_state_dict(torch.load(model_path))

        # prune model
        print()


if __name__ == "__main__":
    args = get_arg_parser()
    args = args.parse_args()
    config = load_config(args.config_path)
    main(config)