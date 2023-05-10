import os
import torch
import wandb
from mlp import MLP
from utils import get_model_path, get_arg_parser, load_config
from loaders import get_loaders
from main_train import main as main_train
from engine_train import evaluate_model


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
sweep_id = wandb.sweep(sweep_config, project="cka_prune")


def main(_config=None):
    with wandb.init(config=_config):
        global CONFIG_PATH
        config = load_config(CONFIG_PATH)
        config["params"]["lr"] = wandb.config.lr
        config["seeds"] = [666]

        # TODO: Set lr in config to lr in wandb_config.
        main_train(config)

        model_path = get_model_path(config["model"], config["seeds"][0])
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
        os.remove(model_path)


if __name__ == "__main__":
    args = get_arg_parser()
    args = args.parse_args()
    CONFIG_PATH = args.config_path
    wandb.agent(sweep_id, main)
