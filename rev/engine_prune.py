import os
import torch
from torch import nn
from loaders import get_loaders
from engine import train_model, evaluate_model
from mlp import MLP, get_activations
from utils import *
from lib.cka import cka, gram_rbf, gram_linear
from torch.nn.utils import prune

# Strategy: one-shot, iterative
# Stopping criteria: iterations, cka
# Type: l1, cka
# multiple layers


def cka_rbf(a: np.array, b: np.array, sigma: float = 1) -> float:
    return cka(
        gram_rbf(a, sigma),
        gram_rbf(b, sigma),
    )


def cka_linear(a: np.array, b: np.array) -> float:
    return cka(
        gram_linear(a),
        gram_linear(b),
    )


def flatten_inner(X):
    outer, *inner = X.shape
    return X.reshape(outer, np.prod(inner))


def get_cka_scores(module, output_act):
    output_act = flatten_inner(output_act)
    output_act_pruned = np.copy(output_act)

    tensor = getattr(module, "weight")
    bias = getattr(module, "bias")

    neurons = tensor.shape[0]
    cka_scores = np.zeros(neurons)

    for i in range(neurons):
        if tensor[i].count_nonzero() == 0:
            continue
        
        neuron_activation = np.copy(output_act[:, i])
        output_act_pruned[:, i] = bias[i]
        cka_scores[i] = cka_linear(output_act, output_act_pruned)
        output_act_pruned[:, i] = neuron_activation
        
    return cka_scores


class SingleNeuronPruningMethod(prune.BasePruningMethod):

    PRUNING_TYPE = "structured"

    def __init__(self, neuron, dim=-1):
        super().__init__()
        self.neuron = neuron
        self.dim = dim

    def compute_mask(self, _, default_mask):
        mask = default_mask.clone()
        mask[self.neuron] = 0
        return mask


def cka_structured(module, module_act, p):
    tensor = module.weight
    # TODO: Replace tensor shape with the number of non-zero neurons.
    prune_count = np.round(p * tensor.shape[0]).astype(int) 

    pruned_neurons = []
    pruned_ckas = []
    for _ in range(prune_count):
        cka_scores = get_cka_scores(module, module_act)
        neuron = cka_scores.argmax()
        SingleNeuronPruningMethod.apply(module, "weight", neuron=neuron)
        pruned_neurons.append(neuron)
        pruned_ckas.append(cka_scores.max())
        print(f"neuron = {pruned_neurons[-1]}, cka = {pruned_ckas[-1]}")
    
    return pruned_neurons, pruned_ckas


def l1_structured(module, module_act, p):
    tensor = module.weight
    # TODO: Same as CKA.
    prune_count = np.round(p * tensor.shape[0]).astype(int)

    normed = torch.linalg.norm(module.weight, dim=-1, ord=1)
    normed[normed == 0] = float('inf')
    neurons = normed.argsort()[:prune_count]

    pruned_module_act = np.copy(module_act)
    pruned_neurons = []
    pruned_ckas = []

    for i in range(neurons.shape[0]):
        neuron = neurons[i].item()
        # cka_scores = get_cka_scores(module, module_act)
        # TODO: Check that this ignores the 0 elements
        SingleNeuronPruningMethod.apply(module, "weight", neuron=neuron)
        pruned_neurons.append(neuron)
        pruned_module_act[:, i] = module.bias[i]
        pruned_ckas.append(cka_linear(module_act, pruned_module_act))
        print(f"neuron = {pruned_neurons[-1]}, cka = {pruned_ckas[-1]}")
    
    return pruned_neurons, pruned_ckas


def prune_one_shot(model, config, prune_func, train_loader, val_loader, test_loader):
    with torch.no_grad():
        model.eval()      

        data = next(iter(val_loader))[0]
        act = get_activations(model, data)

        result = []
        modules = [model.layers[i] for i in config["prune"]["modules"]]
        for module in modules:
            print(f"module = {module}")
            neurons, ckas = prune_func(module, act[module], p=config["prune"]["params"]["rate"])
            _, train_acc = evaluate_model(model, train_loader)
            _, val_acc = evaluate_model(model, val_loader)
            _, test_acc = evaluate_model(model, test_loader)
            result.append({
                "neurons": neurons, 
                "ckas": ckas, 
                "train_acc": train_acc, 
                "val_acc": val_acc,
                "test_acc": test_acc,
            })
        
        return result


def prune_iterative(model, config, prune_func, train_loader, val_loader, test_loader):
    with torch.no_grad():
        model.eval()

        result = []
        for _ in range(config["prune"]["params"]["iterations"]):
            prune_result = prune_one_shot(
                model=model, 
                config=config,
                prune_func=prune_func,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=config["params"]["lr"])
            criterion = nn.CrossEntropyLoss()
            model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=config["params"]["epochs"],
                patience=config["params"]["patience"],
                use_gpu=True
            ).to(torch.device("cpu")) # TODO: Clean this up.

            _, train_acc = evaluate_model(model, train_loader)
            _, val_acc = evaluate_model(model, val_loader)
            _, test_acc = evaluate_model(model, test_loader)
            train_result = {
                "train_acc": train_acc, 
                "val_acc": val_acc,
                "test_acc": test_acc,
            }

            result.append({
                "prune": prune_result,
                "train": train_result,
            })

        return result
