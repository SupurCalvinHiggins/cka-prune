import numpy as np
import torch
from lenet import get_activations
import numpy as np
from lib.cka import cka, gram_rbf, gram_linear


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


def get_cka_scores(module, name, output_act):
    output_act = flatten_inner(output_act)
    output_act_pruned = np.copy(output_act)

    tensor = getattr(module, name)
    bias = getattr(module, 'bias')

    neurons = tensor.shape[0]
    cka_scores = np.zeros(neurons)

    for i in range(neurons):
        if tensor[i].count_nonzero() == 0:
            continue

        neuron_weight = torch.clone(tensor[i])
        neuron_activation = np.copy(output_act[:, i])

        tensor[i] = 0
        output_act_pruned[:, i] = bias[i]

        cka_scores[i] = cka_linear(output_act, output_act_pruned)

        output_act_pruned[:, i] = neuron_activation
        tensor[i] = neuron_weight
    return cka_scores


def cka_structured(model, module, name, data, p=None, n=None, verbose: bool = False):
    assert not (p is None and n is None)
    assert not (p is not None and n is not None)

    tensor = getattr(module, name)

    if p is not None:
        prune_num = np.round(p * tensor.shape[0]).astype(int)
    elif n is not None:
        prune_num = n

    if verbose:
        pruned_neurons = []
        pruned_ckas = []

    tensor = getattr(module, name)
    act = get_activations(model, data)

    for _ in range(prune_num):
        cka_scores = get_cka_scores(module, name, act[module])
        neuron = cka_scores.argmax()
        tensor[neuron] = 0

        if verbose:
            pruned_neurons.append(neuron)
            pruned_ckas.append(cka_scores.max())
    
    if verbose:
        return pruned_neurons, pruned_ckas