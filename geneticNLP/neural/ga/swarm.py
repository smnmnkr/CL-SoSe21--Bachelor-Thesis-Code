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
                noise_tensors[p_id] * score * 100
                for (noise_tensors, score) in noise_tensors_w_score
            ]
        )

        step = learning_rate * (
            (1 / (len(noise_tensors_w_score) * noise_std ** 2)) * sigma
        )

        m_param.data += torch.clamp(step, min=-0.5, max=0.5)
