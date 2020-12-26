from datetime import datetime

import torch

from geneticNLP.data import batch_loader

from geneticNLP.neural.ga import mutate
from geneticNLP.neural.ga.swarm import optimize, optimize_mod

from geneticNLP.neural.ga.utils import evaluate_linear

from geneticNLP.utils import dict_max
from geneticNLP.utils.types import IterableDataset


#
#
#  -------- swarm -----------
#
def swarm(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
    optimizer: str = None,
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

    evaluate_linear(population, train_loader)
    model, _ = dict_max(population)

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        for batch in train_loader:

            noise_tensors_w_score: list = []

            # --- fill new population
            for _ in range(len(population)):

                # created mutated pseudo child
                pseudo_offspring, noise_tensors = mutate(model, noise_std)

                # calculate score
                noise_tensors_w_score.append(
                    [
                        noise_tensors,
                        pseudo_offspring.accuracy(batch),
                    ]
                )

            # --- update model, using custom optimizer
            if optimizer == "custom":
                optimize_mod(
                    model,
                    model.accuracy(batch),
                    noise_tensors_w_score,
                    noise_std,
                    learning_rate,
                )

            # --- update model, using optimizer proposed in Zhang et al.
            else:
                optimize(
                    model,
                    noise_tensors_w_score,
                    noise_std,
                    learning_rate,
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
