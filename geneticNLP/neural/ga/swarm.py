import torch
import torch.nn as nn


from geneticNLP.utils.methods import get_device

#
#
#  -------- optimize -----------
#
def optimize(
    queen: nn.Module,
    swarm: dict,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
):

    #
    scores = torch.tensor(list(swarm.values())).to(get_device())

    #
    swarm_params: list = [
        [param for param in worker.parameters()] for worker in swarm
    ]

    #
    for id_p, q_param in enumerate(queen.parameters()):

        for id_s, s_param in enumerate(
            [worker[id_p] for worker in swarm_params]
        ):
            q_param.data += (
                learning_rate
                / (len(swarm) * noise_std)
                * (s_param * scores[id_s])
            )
