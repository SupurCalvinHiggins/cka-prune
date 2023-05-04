import os
import pickle as pkl
from mlp import MLP
from torch import nn
from loaders import get_loaders
from engine import train_model, evaluate_model
from engine_prune import prune_one_shot, prune_iterative, cka_structured, l1_structured
from utils import *


def main(config_path):
    config = load_config(config_path)
    # TODO: Take result path as arg.
    result_path = get_result_path(random.randint(0, 1_000_000_000))
    print(f"config = {config}")
    print(f"result_path = {result_path}")
    print()

    results = []
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

        # Load the data.
        train_loader, val_loader, test_loader = get_loaders(batch_size=config["params"]["batch_size"])

        # Get the prune function.
        prune_funcs = {
            "cka": cka_structured,
            "l1": l1_structured,
        }
        prune_func = prune_funcs[config["prune"]["params"]["type"]]

        # Get the pruning strategy.
        prune_strategies = {
            "iterative": prune_iterative,
            "one_shot": prune_one_shot,
        }
        prune_strategy = prune_strategies[config["prune"]["strategy"]]

        # Prune the model.
        result = prune_strategy(model, config, prune_func, train_loader, val_loader, test_loader)
        print(result)

        results.append(result)
        with open(result_path, "wb") as f:
            pkl.dump(f, results)
        print()


if __name__ == "__main__":
    args = get_arg_parser()
    args = args.parse_args()
    main(args.config_path)