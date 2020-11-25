from datetime import datetime

import torch

from geneticNLP.data import batch_loader

from geneticNLP.neural.ga import mutate
from geneticNLP.neural.ga.swarm import optimize

from geneticNLP.utils.types import Module, IterableDataset


#
#
#  -------- swarm -----------
#
def swarm(
    model: Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
    population_size: int = 80,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # load dev set as batched loader
    dev_loader = batch_loader(
        dev_set,
        batch_size=batch_size,
    )

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        # load train set as batched loader
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
        )

        for batch in train_loader:

            noise_tensors_w_score: list = []

            # --- fill new population
            for i in range(population_size):

                # created mutated pseudo child
                pseudo_offspring, noise_tensors = mutate(model, noise_std)

                # print("child", i, ":", pseudo_offspring.accuracy(batch))

                # calculate score
                noise_tensors_w_score.append(
                    [noise_tensors, pseudo_offspring.accuracy(batch)]
                )

            # print("model before update: ", model.accuracy(batch))
            # --- update model
            optimize(
                model,
                noise_tensors_w_score,
                noise_std,
                learning_rate,
            )
            noise_tensors_w_score.clear()
            # print(list(model.parameters()))
            # print("model after update: ", model.accuracy(batch))
            # print()
        # exit()

        # --- report
        if epoch % report_rate == 0:

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
