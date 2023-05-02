import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from cka import cka, gram_rbf, gram_linear


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


def compute_heatmap(
    act_a: list[np.array], 
    act_b: list[np.array], 
    cka_func=cka_rbf
) -> np.array:

    act_a = [a.mean(axis=0) for a in act_a]
    act_b = [b.mean(axis=0) for b in act_b]
    
    heatmap = np.zeros((len(act_a), len(act_b)))
    for i, a in enumerate(act_a):
        for j, b in enumerate(act_b):
            heatmap[i][j] = cka_func(a, b)
    return heatmap


def display_heatmap(heatmap: np.array) -> None:
    sns.set()
    ax = sns.heatmap(heatmap, vmin=0, vmax=1)
    ax.invert_yaxis()
    plt.show()