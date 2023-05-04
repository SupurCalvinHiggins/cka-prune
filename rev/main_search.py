import torch
import wandb
from mlp import MLP
from utils import get_model_path
from loaders import get_loaders
from main_train import main as main_train
from engine import evaluate_model


wandb.login()
sweep_config = {
    "method": "grid",
    "metric": {
        "name": "val_loss",
        "goal": "minimize",
    },
    "parameters": {
        "lr": {
            "values": [0.1, 0.01, 0.001, 0.0001, 0.00001],
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="sweep-784-1024-1024-10")


def main(_config=None):
    with wandb.init(config=_config):
        # TODO: Set lr in config to lr in wandb_config.
        config = {
            "model": {
                "input_size": 784,
                "hidden_sizes": [1024, 1024],
                "output_size": 10,
                "dropout_rate": 0.5
            },
            "params": {
                "lr": wandb.config.lr,
                "batch_size": 512,
                "epochs": 50,
                "patience": 3
            },
            "seeds": [0]
        }
        main_train(config)

        model_path = get_model_path(config["model"], config["seed"][0])
        model = MLP(
            input_size=config["model"]["input_size"],
            hidden_sizes=config["model"]["hidden_sizes"],
            output_size=config["model"]["output_size"],
            dropout_rate=config["model"]["dropout_rate"],
        )
        model.load_state_dict(torch.load(model_path))
        _, val_loader, _ = get_loaders(config["params"]["batch_size"])
        criterion = torch.nn.CrossEntropyLoss()
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        wandb.log({"val_loss": val_loss, "val_acc": val_acc})


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
