# Representation Aware Pruning with Centered Kernel Alignment

Some experiments with a new structured pruning algorithm based on centered kernel alignment. Created for my CSC 561 final project at the University of Rhode Island.

## Overview

Centered kernel alignment (CKA) provides a measure of similiarity between neural network activations. CKA can be used to compute the relative "damage" caused by pruning a particular neuron via the following procedure.

For each neuron in the layer
1. Set the weights of the neuron to zero.
2. Compute the layer activations.
3. Compute the CKA score between the original and new activations.
4. Restore the weights of the neuron.

Initutively, high CKA scores correspond to low damage to network representations. Conversely, low CKA scores correspond to high damage to network representations. 

To prune the network, neurons are greedily removed from each layer according to the relative damage scores until all layers have been pruned by $p$%. The damages scores are recomputed after removing each neuron and the original activations are recomputed at the start of each pruning pass. This takes $\Omega(n^2)$ time where $n$ is the number of neurons.

This repository contains scripts for reproducing all project results. This includes the hyperparameter search, model training and pruning. 

## Installation

### Python

This project requires Python 3.9.6 or Python 3.10.6. Other Python version might also work but have not been tested. See [here](https://www.python.org/downloads/) for details on installing Python 3.9.6 or Python 3.10.6.

### Python Dependencies

The Python dependencies are listed in __requirements.txt__. Installation instructions for these dependencies are given below.

Navigate to the root directory of the project and execute the following command.

```bash
pip install -r requirements.txt
```
