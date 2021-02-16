import torch
import torch.nn as nn


from beyondGD.utils.methods import get_device, smooth_gradient

#
#
#  -------- optimize -----------
#
def optimize(
    model: nn.Module,
    model_score: float,
    noise_tensors_w_score: list,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
    filter: bool = True,
):

    # --- filter by higher scoring noise tensors (optional)
    if filter:
        noise_tensors_w_score: list = [
            (noise_tensors, score)
            for noise_tensors, score in noise_tensors_w_score
            if score > model_score
        ]

        if len(noise_tensors_w_score) == 0:
            return model

    #
    for p_id, m_param in enumerate(model.parameters()):

        sigma = torch.empty(m_param.shape).to(get_device())

        sigma += sum(
            [
                noise_tensors[p_id]
                * score  # center score: (score - model_score)
                for (
                    noise_tensors,
                    score,
                ) in noise_tensors_w_score
            ]
        )

        step = learning_rate * (
            (1 / (len(noise_tensors_w_score) * noise_std ** 2))
            * sigma
        )

        m_param.data += smooth_gradient(step)

    return model