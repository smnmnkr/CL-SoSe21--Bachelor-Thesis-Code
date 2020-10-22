import copy
import torch

#
#
#  -------- mutate -----------
#
@torch.no_grad()
def mutate(
    parent_network,
    mutation_rate: float = 0.02,
):

    child_network = copy.deepcopy(parent_network)

    for param in child_network.parameters():

        # mut_tensor = torch.randn_like(param)
        mut_tensor = torch.empty(param.shape).normal_(
            mean=0,
            std=mutation_rate,
        )

        param.data += mut_tensor

    return child_network
