from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
        input_size: int, 
        hidden_sizes: list[int], 
        output_size: int, 
        dropout_rate: float
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.layers = nn.ParameterList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(F.relu(layer(x)))
        return self.output_layer(x)


def get_activation_hook(act, idx):
    def hook(model, input, output):
        act[idx] = output.detach().numpy()
    return hook


def add_hooks(model):
    act = {}
    handles = [
        layer.register_forward_hook(get_activation_hook(act, layer))
        for layer in model.layers
    ]
    return act, handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def get_activations(model, data):
    act, handles = add_hooks(model)
    _ = model(data)
    remove_hooks(handles)
    return act