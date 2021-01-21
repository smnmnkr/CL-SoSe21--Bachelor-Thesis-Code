import copy
import torch

from beyondGD.utils import get_device

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
    noise_tensors: list = []

    for param in child_network.parameters():

        mut_tensor = (
            torch.empty(param.shape)
            .normal_(
                mean=0,
                std=mutation_rate,
            )
            .to(get_device())
        )

        noise_tensors.append(mut_tensor)
        param.data += mut_tensor

    return child_network, noise_tensors
