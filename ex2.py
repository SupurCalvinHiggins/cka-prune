#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.utils.prune as prune
from lenet import LeNet, get_activations
from heatmap import compute_heatmap, display_heatmap, cka_linear
from loaders import get_mnist_loaders
from prune import cka_structured
import pickle as pkl


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


def compute_acc(model, loader):
    model.eval()
    correct = 0
    for _, (batch_x, batch_y) in enumerate(loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred_y = torch.argmax(model(batch_x), axis=-1)
        correct += torch.sum(torch.eq(pred_y, batch_y))
    return (correct / len(loader.dataset)).item()


# In[ ]:


def cka_structured_all(model, module, data, cka_threshold=None): # 0.995
    model.eval()
    with torch.no_grad():
        result = {"cka": [], "val_acc": [], "pruned": []}

        org_weight = torch.clone(module.weight)
        _, val_loader = get_mnist_loaders(60)

        neurons = module.weight.shape[0]
        for i in range(neurons):
            module, pruned, ckas = cka_structured(
                model, module, 'weight', data, n=1, verbose=True)
            val_acc = compute_acc(model, val_loader)

            result["cka"].append(ckas[0])
            result["pruned"].append(pruned[0])
            result["val_acc"].append(val_acc)

            print(f"Pruned neuron {pruned[0]} with CKA = {ckas[0]}, val_acc = {val_acc} and p = {(i + 1) / neurons}")
            if cka_threshold is not None and ckas[0] < cka_threshold:
                break
        
        module.weight[:, :] = org_weight

    return result


# In[ ]:


def l1_structured_all(model, module, data):
    model.eval()
    with torch.no_grad():
        result = {"cka": [], "val_acc": [], "pruned": []}

        act = get_activations(model, data)
        neuron_order = torch.argsort(torch.linalg.vector_norm(torch.clone(module.weight), ord=1, dim=-1))

        neurons = module.weight.shape[0]
        for i in range(neurons):
            prune.ln_structured(module, 'weight', amount=1, n=1, dim=0)

            pruned_act = get_activations(model, data)
            _, val_loader = get_mnist_loaders(60)
            val_acc = compute_acc(model, val_loader)
            cka = cka_linear(act[module], pruned_act[module])

            result["cka"].append(cka)
            result["pruned"].append(neuron_order[i].item())
            result["val_acc"].append(val_acc)

            print(f"Pruned with L1 norm with CKA = {cka}, val_acc = {val_acc} and p = {(i + 1) / neurons}")
    return result


# In[ ]:


prune_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for FC0_PRUNE_RATE in prune_rates:
    for FC1_PRUNE_RATE in prune_rates:
        DROPOUT_RATE = 0.5
        DROPOUT_RATE_STR = str(int(DROPOUT_RATE * 100)).zfill(2)
        FC0_PRUNE_RATE_STR = str(int(FC0_PRUNE_RATE * 100)).zfill(2)
        FC1_PRUNE_RATE_STR = str(int(FC1_PRUNE_RATE * 100)).zfill(2)

        print(f"FC0_PRUNE_RATE = {FC0_PRUNE_RATE}, FC1_PRUNE_RATE = {FC1_PRUNE_RATE}")

        results = []
        for i in range(30):
            
            # Load model.
            model = LeNet('0d', DROPOUT_RATE)
            model_path = f"models/lenet-0d-{DROPOUT_RATE_STR}-{i}.model"
            model.load_state_dict(torch.load(model_path))

            # Prune with CKA.
            train_loader, val_loader = get_mnist_loaders(60)
            data = next(iter(train_loader))[0].to(device)

            model.eval()
            with torch.no_grad():
                _, fc0_neurons, fc0_ckas = cka_structured(model, model.fc0, 'weight', data, p=FC0_PRUNE_RATE, verbose=True)
                _, fc1_neurons, fc1_ckas = cka_structured(model, model.fc1, 'weight', data, p=FC1_PRUNE_RATE, verbose=True)
                val_acc = compute_acc(model, val_loader)
            
            fc0_fc1_result = {
                "fc0": {"pruned": fc0_neurons, "cka": fc0_ckas},
                "fc1": {"pruned": fc1_neurons, "cka": fc1_ckas},
                "val_acc": val_acc,
            }

            # Reload model.
            model = LeNet('0d', DROPOUT_RATE)
            model.load_state_dict(torch.load(model_path))

            model.eval()
            with torch.no_grad():
                _, fc1_neurons, fc1_ckas = cka_structured(model, model.fc1, 'weight', data, p=FC1_PRUNE_RATE, verbose=True)
                _, fc0_neurons, fc0_ckas = cka_structured(model, model.fc0, 'weight', data, p=FC0_PRUNE_RATE, verbose=True)
                val_acc = compute_acc(model, val_loader)

            fc1_fc0_result = {
                "fc0": {"pruned": fc0_neurons, "cka": fc0_ckas},
                "fc1": {"pruned": fc1_neurons, "cka": fc1_ckas},
                "val_acc": val_acc,
            }

            # Update results.
            results.append({"fc0_fc1": fc0_fc1_result, "fc1_fc0": fc1_fc0_result})
            print(results)

            # Save results
            output_path = f"output/ex2/output-{FC0_PRUNE_RATE_STR}-{FC1_PRUNE_RATE_STR}.pkl"
            with open(output_path, 'wb') as f:
                pkl.dump(results, f)

