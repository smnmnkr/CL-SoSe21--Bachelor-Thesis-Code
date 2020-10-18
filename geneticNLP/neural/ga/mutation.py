import copy
import torch

#
#
#  -------- mutate -----------
#
@torch.no_grad()
def mutate(
    parent_network,
    score: float = 0.0,
    mutation_rate: float = 0.02,
):

    child_network = copy.deepcopy(parent_network)

    for param in child_network.parameters():
        param.data += (
            mutation_rate * torch.randn_like(param) * (1.0 - score)
        )

    return child_network
