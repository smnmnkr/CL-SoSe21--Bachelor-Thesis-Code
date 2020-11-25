from datetime import datetime

import torch

from geneticNLP.data import batch_loader

from geneticNLP.neural.ga.swarm import optimize
from geneticNLP.neural.ga.utils import (
    evaluate_linear,
    process_linear,
)

from geneticNLP.utils import get_device
from geneticNLP.utils.types import Module, IterableDataset


#
#
#  -------- swarm -----------
#
def swarm(
    model_CLS: Module,
    config: dict,
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

    # generate queen, base population
    queen: Module = model_CLS(config).to(get_device())
    population: dict = {
        model_CLS(config).to(get_device()): 0.0
        for _ in range(population_size)
    }

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        # load train set as batched loader
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
        )

        # --- if is first epoch evaluate models at first
        if epoch == 1:
            evaluate_linear(population, train_loader)

        for batch in train_loader:

            # --- process generation
            population = process_linear(
                population,
                batch,
                population_size=population_size,
                selection_rate=1,
                crossover_rate=0.0,
            )

            # --- update queen model
            optimize(
                queen,
                population,
                noise_std,
                learning_rate,
            )

        # --- report
        if epoch % report_rate == 0:

            # --- evaluate all models on train set
            evaluate_linear(population, train_loader)

            print(
                "[--- @{:02}: \t avg(train)={:2.4f} \t queen(train)={:2.4f} \t queen(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    sum(population.values()) / len(population),
                    queen.evaluate(train_loader),
                    queen.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    # --- return queen
    return queen
