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


def get_cka_scores(module, name, input_act, output_act):
    output_act = flatten_inner(output_act)
    tensor = getattr(module, name)
    neurons = tensor.shape[0]
    cka_scores = np.zeros(neurons)
    for i in range(neurons):
        if tensor[i].count_nonzero() == 0:
            continue

        neuron_weight = torch.clone(tensor[i])
        tensor[i] = 0
        output_act_pruned = flatten_inner(module(torch.tensor(input_act))).detach().numpy()
        tensor[i] = neuron_weight
        cka_scores[i] = cka_linear(output_act, output_act_pruned)
    return cka_scores


def cka_structured(model, module, name, data, p):
    tensor = getattr(module, name)
    prune_num = np.floor(p * tensor.shape[0]).astype(int)
    input_act, output_act = get_activations(model, module, data)
    for _ in range(prune_num):
        cka_scores = get_cka_scores(module, name, input_act, output_act)
        neuron = cka_scores.argmax()
        tensor[neuron] = 0
    return module