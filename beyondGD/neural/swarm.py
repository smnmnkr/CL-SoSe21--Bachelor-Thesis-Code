from datetime import datetime

import torch
import torch.nn as nn

from beyondGD.data import batch_loader

from beyondGD.neural.ga import mutate
from beyondGD.neural.ga.swarm import optimize


from beyondGD.utils.types import IterableDataset


#
#
#  -------- swarm -----------
#
def swarm(
    model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
    population_size: int = 200,
    filter_offspring: bool = False,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # load train set as batched loader
    train_loader = batch_loader(
        train_set,
        batch_size=batch_size,
    )

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        for batch in train_loader:

            noise_tensors_w_score: list = []

            # --- fill new population
            for _ in range(population_size):

                # created mutated pseudo child
                pseudo_offspring, noise_tensors = mutate(
                    model, noise_std
                )

                # calculate score
                noise_tensors_w_score.append(
                    [
                        noise_tensors,
                        pseudo_offspring.accuracy(batch),
                    ]
                )

            # --- update model, using optimizer proposed in Zhang et al. (with optional filtering)
            model = optimize(
                model,
                noise_tensors_w_score,
                noise_std,
                learning_rate,
                filter=filter_offspring,
            )

        # --- report
        if epoch % report_rate == 0:

            # load dev set as batched loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
            )

            print(
                "[--- @{:02}: \t acc(train)={:2.4f} \t acc(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    model.evaluate(train_loader),
                    model.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    # --- return model
    return model
