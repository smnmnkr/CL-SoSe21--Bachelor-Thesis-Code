import random

from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from geneticNLP.neural.ga import mutate, elitism
from geneticNLP.neural.ga.swarm import optimize

from geneticNLP.data import batch_loader
from geneticNLP.utils.methods import get_device


#
#
#  -------- hybrid -----------
#
def hybrid(
    model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
    convergence_min: int = 0.8,
    population_size: int = 80,
    selection_rate: float = 10,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # load dev set as batched loader
    dev_loader = batch_loader(
        dev_set,
        batch_size=batch_size,
        num_workers=0,
    )

    # start convergence, epoch
    convergence: float = 0.0
    epoch: int = 0

    # generate queen, swarm
    queen: nn.Module = model
    swarm: dict = {}

    # --
    while convergence < convergence_min:
        time_begin = datetime.now()

        # load train set as batched loader
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
            num_workers=0,
        )

        for batch in train_loader:

            # --- selection if is not first epoch else use queen
            selection: dict = (
                elitism(swarm, selection_rate)
                if (epoch > 0)
                else {queen: queen.accuracy(batch)}
            )
            swarm.clear()

            # --- mutation
            for _ in range(population_size):

                # get random entitiy from selection
                rnd_entitiy, score = random.choice(list(selection.items()))

                mut_entitiy = mutate(rnd_entitiy, 1 - score)

                swarm[mut_entitiy] = mut_entitiy.accuracy(batch)

            # --- update queen model
            optimize(
                queen,
                swarm,
                noise_std,
                learning_rate,
            )

        # --- increase epoch
        epoch += 1

        # --- report
        if (epoch + 1) % report_rate == 0:
            convergence = queen.evaluate(train_loader)

            print(
                "[--- @{:02}: \t swarm(train)={:2.4f} \t queen(train)={:2.4f} \t queen(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    (epoch + 1),
                    sum(swarm.values()) / len(swarm),
                    convergence,
                    queen.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )
