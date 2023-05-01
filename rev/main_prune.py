import os
import torch
import argparse
import pickle as pkl
from torch import nn
from lenet import LeNet
from loaders import get_loaders
from engine import train_model, evaluate_model
from utils import *
from prune import cka_structured
from torch.nn.utils.prune import ln_structured


def get_arg_parser():
    parser = argparse.ArgumentParser('LeNet-300-100 Training')

    # Training configuration.
    parser.add_argument('--model_count', type=int, required=True)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--use_gpu', default=False, type=bool)

    # Hyper-parameters.
    parser.add_argument('--lr', default=0.0012, type=float)
    parser.add_argument('--batch_size', default=60, type=int)
    # parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--use_1d', default=False, type=bool)

    # Modules to prune.
    parser.add_argument('--modules', nargs='+', default=['fc0', 'fc1'])

    # Pruning strategy.
    parser.add_argument('--type', choices=['cka', 'l1'], default='cka', type=str)
    parser.add_argument('--iters', default=15, type=int)
    parser.add_argument('--percent', default=0.2, type=float)

    # Which one.
    parser.add_argument('--seed', required=True, type=int)

    return parser


def main(args):
    print(args)
    print()

    for seed in range(args.seed, args.seed + args.model_count):

        # Set seed.
        print("*** PRUNING ***")
        print(f"seed = {seed}")
        set_seed(seed)

        # Skip models that do not exist.
        base_model_path = get_model_path(seed, args.dropout_rate, args.use_1d)
        if not os.path.exists(base_model_path):
            print("*** FAILED ***")
            print("model does not exist")
            print(f"base_model_path = {base_model_path}")
            continue
        
        # Set up base model.
        dropout = nn.Dropout if not args.use_1d else nn.Dropout1d
        model = LeNet(dropout, args.dropout_rate)
        model.load_state_dict(torch.load(base_model_path))

        modules = [getattr(model, module_name) for module_name in args.modules]

        # Get data loaders.
        train_loader, val_loader, test_loader = get_loaders(args.batch_size, args.use_gpu)
        
        p = args.percent
        q = 1 - p
        output = []
        # repeat for some number of iterations
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for i in range(args.iters):
            print()
            print(f"i = {i}")

            _, before_prune_val_acc = evaluate_model(model, val_loader, criterion)
            _, before_prune_test_acc = evaluate_model(model, test_loader, criterion)

            print("before pruning")
            print(f"val_acc = {before_prune_val_acc}, test_acc = {before_prune_test_acc}")
            for module in modules:
                pruned_count = torch.count_nonzero(module.weight, dim=-1).eq(0).sum()
                print(f"module = {module}, pruned_count = {pruned_count}")

            data = next(iter(train_loader))[0]
            p_i = p * (q ** i)
            if args.type == 'cka':
                pruned_neurons_and_ckas = []
                for module in modules:
                    pruned_neurons_and_ckas.append(cka_structured(model, module, 'weight', data, p=p_i, verbose=True))
            else:
                pruned_neurons_and_ckas = []
                for module in modules:
                    ln_structured(module, 'weight', p, n=1, dim=0)

            _, after_prune_val_acc = evaluate_model(model, val_loader, criterion)
            _, after_prune_test_acc = evaluate_model(model, test_loader, criterion)

            print()
            print("after pruning")
            print(f"val_acc = {before_prune_val_acc}, test_acc = {before_prune_test_acc}")
            for module in modules:
                pruned_count = torch.count_nonzero(module.weight, dim=-1).eq(0).sum()
                print(f"module = {module}, pruned_count = {pruned_count}")
            print(f"pruned_neurons_and_ckas = {pruned_neurons_and_ckas}")

            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=args.epochs,
                patience=args.patience,
                use_gpu=args.use_gpu,
            )

            _, after_train_val_acc = evaluate_model(model, val_loader, criterion)
            _, after_train_test_acc = evaluate_model(model, test_loader, criterion)

            print()
            print("after retraining")
            print(f"val_acc = {after_train_val_acc}, test_acc = {after_train_test_acc}")

            iter_output = {
                "pruned_neurons_and_ckas": pruned_neurons_and_ckas,
                "before_prune": {
                    "val_acc": before_prune_val_acc,
                    "test_acc": before_prune_test_acc,
                },
                "after_prune": {
                    "val_acc": after_prune_val_acc,
                    "test_acc": after_prune_test_acc,
                },
                "after_train": {
                    "val_acc": after_train_val_acc,
                    "test_acc": after_train_test_acc,
                },
            }
            output.append(iter_output)

            checkpoint_path = (
                f"models/"
                f"pruned-lenet-300-100"
                f"_seed-{seed}"
                f"_p-{p}"
                f"_i-{i}"
                f"_type-{args.type}"
                f"_module-{'-'.join(args.modules)}"
                f".pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
        
        result = {
            "output": output,
            "args": args, 
        }
        result_path = (
            f"output/"
            f"pruned-lenet-300-100"
            f"_seed-{seed}"
            f"_p-{p}"
            f"_type-{args.type}"
            f"_module-{'-'.join(args.modules)}"
            f".pkl"
        )
        with open(result_path, 'wb') as f:
            pkl.dump(result, f)


if __name__ == "__main__":
    args = get_arg_parser()
    args = args.parse_args()
    main(args)