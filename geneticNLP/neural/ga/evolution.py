import random, operator, multiprocessing

from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from geneticNLP.neural.ga import mutate, elitism, cross

from geneticNLP.data import batch_loader

# prevent MAC OSX multiprocessing bug
# src: https://github.com/pytest-dev/pytest-flask/issues/104
multiprocessing.set_start_method("fork")

#
#
#  -------- evolve -----------
#
def evolve(
    model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    mutation_rate: float = 0.02,
    convergence_min: int = 0.95,
    population_size: int = 80,
    survivor_rate: float = 10,
    crossover_rate: float = 0.5,
    report_rate: int = 50,
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

    # generate base population
    population: dict = {model: 0.0}

    # --
    while convergence < convergence_min:
        time_begin = datetime.now()

        # load train set as batched loader
        # TODO: enable adaptive batch size
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
            num_workers=0,
        )

        for batch in train_loader:

            # --- select by elite if is not first epoch else use only input model
            selection: dict = (
                elitism(population, survivor_rate)
                if (epoch > 0)
                else population
            )

            # --- add selection to next generation
            next_generation: list = [
                selected for selected, _ in selection.items()
            ]

            # --- mutation
            for _ in range(population_size - survivor_rate):

                # get random players from selection
                random_selected_1, _ = random.choice(
                    list(selection.items())
                )
                random_selected_2, _ = random.choice(
                    list(selection.items())
                )

                # (optionally) cross random players
                if crossover_rate > random.uniform(0, 1) and epoch > 0:
                    random_selected_1 = cross(
                        random_selected_1, random_selected_2
                    )

                # mutate random selected
                random_mutated = mutate(random_selected_1, mutation_rate)

                # mutate and append it
                next_generation.append(random_mutated)

            population = dict.fromkeys(next_generation, 0.0)

            # --- evaluate score on batch
            for item, _ in population.items():
                population[item] = item.accuracy(batch)

        # --- increase epoch, get best
        epoch += 1

        # --- report
        if (epoch + 1) % report_rate == 0:

            # --- evaluate all models on train set
            for item, _ in population.items():

                def worker_eval(model):
                    population[model] = model.evaluate(train_loader)

                processes = multiprocessing.Process(
                    target=worker_eval, args=(item,)
                )

                processes.start()

            # --- find best model and corresponding score
            best, score = max(
                population.items(), key=operator.itemgetter(1)
            )
            convergence = score

            print(
                "[--- @{:02}: \t acc(train)={:2.4f} \t acc(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    (epoch + 1),
                    score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )
