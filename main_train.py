import os
import torch
from torch import nn
from loaders import get_loaders
from engine_train import train_model
from mlp import MLP
from utils import *


def main(config):
    print(config)
    print()

    for seed in config["seeds"]:

        # Set seed.
        print("*** Training ***")
        print(f"seed = {seed}")
        set_seed(seed)

        # Get model path.
        model_path = get_model_path(config["model"], seed)
        print(f"model_path = {model_path}")

        # Skip models that already exist.
        if os.path.exists(model_path):
            print("model already exists")
            print()
            continue
            
        # Set up model.
        model = MLP(
            input_size=config["model"]["input_size"],
            hidden_sizes=config["model"]["hidden_sizes"],
            output_size=config["model"]["output_size"],
            dropout_rate=config["model"]["dropout_rate"],
        )

        # Get data loaders.
        train_loader, val_loader, _ = get_loaders(config["params"]["batch_size"])

        # Set up optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=config["params"]["lr"])
        criterion = nn.CrossEntropyLoss()

        # Train model.
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config["params"]["epochs"],
            patience=config["params"]["patience"],
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