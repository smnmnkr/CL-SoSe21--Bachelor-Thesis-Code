import torch
import torch.nn as nn


from geneticNLP.utils.methods import get_device

#
#
#  -------- optimize -----------
#
def optimize(
    model: nn.Module,
    noise_tensors_w_score: list,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
):

    #
    for p_id, m_param in enumerate(model.parameters()):

        sigma = torch.empty(m_param.shape).to(get_device())

        sigma += sum(
            [
                noise_tensors[p_id] * score
                for (noise_tensors, score) in noise_tensors_w_score
            ]
        )

        step = learning_rate * (
            (1 / (len(noise_tensors_w_score) * noise_std ** 2)) * sigma
        )

        # possible problem: vanishing gradient
        # solution: set nan tensors to zero
        # src: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918
        step[step != step] = 0.0

        # possible problem: exploding gradient
        # solution: clamp all into the range [ min, max ]
        # src: https://pytorch.org/docs/stable/generated/torch.clamp.html
        torch.clamp(step, min=-60.0, max=60.0)

        m_param.data += step
