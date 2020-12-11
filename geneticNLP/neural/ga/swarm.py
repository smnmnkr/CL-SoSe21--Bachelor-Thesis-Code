import torch
import torch.nn as nn


from geneticNLP.utils.methods import get_device, smooth_gradient

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
                for (
                    noise_tensors,
                    score,
                ) in noise_tensors_w_score
            ]
        )

        step = learning_rate * (
            (1 / (len(noise_tensors_w_score) * noise_std ** 2)) * sigma
        )

        m_param.data += smooth_gradient(step)


#
#
#  -------- optimize_mod -----------
#
def optimize_mod(
    model: nn.Module,
    model_score: float,
    noise_tensors_w_score: list,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
):

    filtered_noise_tensors_w_score: list = [
        (noise_tensors, score)
        for noise_tensors, score in noise_tensors_w_score
        if score > model_score
    ]

    #
    for p_id, m_param in enumerate(model.parameters()):

        sigma = torch.empty(m_param.shape).to(get_device())

        sigma += sum(
            [
                noise_tensors[p_id] * score
                for (
                    noise_tensors,
                    score,
                ) in filtered_noise_tensors_w_score
            ]
        )

        if len(filtered_noise_tensors_w_score) == 0:
            return

        step = learning_rate * (
            (1 / (len(filtered_noise_tensors_w_score) * noise_std ** 2))
            * sigma
        )

        m_param.data += smooth_gradient(step)
