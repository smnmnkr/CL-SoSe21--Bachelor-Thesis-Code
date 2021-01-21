from datetime import datetime

import torch

from beyondGD.data import batch_loader
from beyondGD.utils import dict_max

from beyondGD.neural.ga.utils import (
    evaluate_linear,
    process_linear,
)

from beyondGD.utils.types import IterableDataset


#
#
#  -------- evolve -----------
#
def evolve(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    selection_rate: int = 10,
    crossover_rate: float = 0.5,
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

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        for batch in train_loader:

            # --- process generation
            population = process_linear(
                population,
                batch,
                selection_rate=selection_rate,
                crossover_rate=crossover_rate,
            )

        # --- report
        if epoch % report_rate == 0:

            # --- evaluate all models on train set
            evaluate_linear(population, train_loader)

            # --- find best model and corresponding score
            best, score = dict_max(population)

            # load dev set as batched loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
                num_workers=0,
            )

            print(
                "[--- @{:02}: \t avg(train)={:2.4f} \t best(train)={:2.4f} \t best(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    sum(population.values()) / len(population),
                    score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    return population
