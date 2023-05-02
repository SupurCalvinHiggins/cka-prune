import os
import torch
import argparse
from torch import nn
from loaders import get_loaders
from engine import train_model
from mlp import MLP
from utils import *


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    return parser


def main(config):
    print(config)
    print()

    for seed in config["seeds"]:

        # Set seed.
        print("*** Training ***")
        print(f"seed = {seed}")
        set_seed(seed)

        # Set up model.
        model = MLP(
            input_size=config["input_size"],
            hidden_sizes=config["hidden_sizes"],
            output_size=config["output_size"],
            dropout_rate=config["dropout_rate"],
        )

        # Skip models that already exist.
        model_path = get_model_path(model, seed)
        if os.path.exists(model_path):
            print("model already exists")
            print()
            continue

        # Get data loaders.
        train_loader, val_loader, _ = get_loaders(config["batch_size"])

        # Set up optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()

        # Train model.
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config["epochs"],
            patience=config["patience"],
            use_gpu=True,
        )

        # Save model.
        torch.save(model.state_dict(), model_path)
        print()
   

if __name__ == "__main__":
    args = get_arg_parser()
    args = args.parse_args()
    config = load_config(args.config_path)
    main(config)