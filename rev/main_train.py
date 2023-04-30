import os
import torch
import argparse
from torch import nn
from lenet import LeNet
from loaders import get_loaders
from engine import train_model
from utils import *


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

    return parser


def main(args):
    print(args)
    print()

    for seed in range(args.model_count):

        # Set seed.
        print("*** Training ***")
        print(f"seed = {seed}")
        set_seed(seed)

        # Skip models that already exist.
        model_path = get_model_path(seed, args.dropout_rate, args.use_1d)
        if os.path.exists(model_path):
            print("model already exists")
            print()
            continue
        
        # Set up model.
        dropout = nn.Dropout if not args.use_1d else nn.Dropout1d
        model = LeNet(dropout, args.dropout_rate)

        # Get data loaders.
        train_loader, val_loader, test_loader = get_loaders(args.batch_size)

        # Set up optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Train model.
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=args.epochs,
            patience=args.patience,
            use_gpu=args.use_gpu,
        )

        # Save model.
        torch.save(model.state_dict(), model_path)
        print()
   

if __name__ == "__main__":
    args = get_arg_parser()
    args = args.parse_args()
    main(args)