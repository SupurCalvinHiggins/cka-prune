import os
import torch
import argparse
from torch import nn
from lenet import LeNet
from loaders import get_loaders
from engine import train_model, evaluate_model
from utils import *
from prune import cka_structured


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
    parser.add_argument('--modules', nargs='+', default=[])

    # Pruning strategy.
    parser.add_argument('--iterative', default=False, type=bool)
    parser.add_argument('--type', choices=['cka', 'l1'])

    return parser


def main(args):
    print(args)
    print()

    for seed in range(args.model_count):

        # Set seed.
        print("*** PRUNING ***")
        print(f"seed = {seed}")
        set_seed(seed)

        # Skip models that do not exist.
        model_path = get_model_path(seed, args.dropout_rate, args.use_1d)
        if not os.path.exists(model_path):
            print("*** FAILED ***")
            print("model does not exist")
            print(f"model_path = {model_path}")
            continue
        
        # Set up model.
        dropout = nn.Dropout if not args.use_1d else nn.Dropout1d
        model = LeNet(dropout, args.dropout_rate)
        model.load_state_dict(torch.load(model_path))

        # Get data loaders.
        train_loader, val_loader, test_loader = get_loaders(args.batch_size)
        
        p = 0.2
        q = 1 - p
        iters = 10
        output = []
        # repeat for some number of iterations
        for i in range(iters):
            
            data = next(iter(train_loader))[0]
            # ensure that this percentage shrinks with number of iterations
            p_i = p * (q ** i)
            pruned_neurons, pruned_ckas = cka_structured(model, model.fc0, 'weight', data, p=p_i, verbose=True)

            # call train model
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()
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

            _, val_acc = evaluate_model(model, val_loader, criterion)
            _, test_acc = evaluate_model(model, test_loader, criterion)

            iter_output = {
                "pruned_neurons": pruned_neurons,
                "pruned_ckas": pruned_ckas,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
            # save evaluation
            output.append(iter_output)
            print(output)

        print()


if __name__ == "__main__":
    args = get_arg_parser()
    args = args.parse_args()
    main(args)