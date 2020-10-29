from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from geneticNLP.data import batch_loader
from geneticNLP.utils import dict_max

from geneticNLP.neural.ga.utils import (
    evaluate_parallel,
    process_non_parallel,
)


#
#
#  -------- evolve -----------
#
def evolve(
    model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    convergence_min: int = 0.8,
    population_size: int = 80,
    selection_rate: int = 10,
    crossover_rate: float = 0.5,
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

    # start convergence, epoch, population
    convergence: float = 0.0
    epoch: int = 0
    population: dict = {model: model.evaluate(dev_loader)}

    # --
    while convergence < convergence_min:
        epoch += 1
        time_begin = datetime.now()

        # load train set as batched loader
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
        )

        for batch in train_loader:

            # --- process generation
            population = process_non_parallel(
                population,
                batch,
                population_size=population_size,
                selection_rate=selection_rate,
                crossover_rate=crossover_rate,
            )

        # --- report
        if (epoch + 1) % report_rate == 0:

            # --- evaluate all models on train set
            evaluate_parallel(population, train_loader)

            # --- find best model and corresponding score
            best, score = dict_max(population)
            convergence = score

            print(
                "[--- @{:02}: \t avg(train)={:2.4f} \t best(train)={:2.4f} \t best(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    (epoch + 1),
                    sum(population.values()) / len(population),
                    score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )
