import numpy as np
import torch
from heatmap import cka_linear


def flatten_inner(X):
    outer, *inner = X.shape
    return X.reshape(outer, np.prod(inner))


def get_activations(model, module, data):
    input_act = None
    output_act = None

    def hook(model, input, output):
        nonlocal input_act, output_act
        input_act = input[0].detach().numpy()
        output_act = output.detach().numpy()
    
    handle = module.register_forward_hook(hook)
    _ = model(torch.tensor(data))
    handle.remove()

    return input_act, output_act


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

    if p is not None:
        prune_num = np.round(p * tensor.shape[0]).astype(int)
    elif n is not None:
        prune_num = n

    if verbose:
        neurons = []
        ckas = []

    tensor = getattr(module, name)
    _, output_act = get_activations(model, module, data)
    
    for _ in range(prune_num):
        cka_scores = get_cka_scores(module, name, output_act)
        neuron = cka_scores.argmax()
        tensor[neuron] = 0

        if verbose:
            neurons.append(neuron)
            ckas.append(cka_scores.max())
    
    if verbose:
        return module, neurons, ckas

    return module