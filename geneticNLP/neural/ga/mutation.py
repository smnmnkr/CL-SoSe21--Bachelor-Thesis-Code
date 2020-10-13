import random
import copy

import torch

#
#
#  -------- mutate -----------
#
def mutate(
    parent_network,
    mutation_rate: float = 0.02,
):

    child_network = copy.deepcopy(parent_network)

    for layer in child_network.parameters():
        layer = layer + torch.rand(tuple(layer.shape))  # * mutation_rate

    return child_network
