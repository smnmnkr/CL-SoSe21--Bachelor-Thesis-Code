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
    selection_rate: float = 10,
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

    # start convergence, epoch, population
    convergence: float = 0.0
    epoch: int = 0
    population: dict = {}

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

            # --- select by elite if is not first epoch else use only input model
            selection: dict = (
                elitism(population, selection_rate)
                if (epoch > 0)
                else {model: 0.0}
            )
            population.clear()

            # --- mutation
            for _ in range(population_size):

                # get random players from selection
                rnd_entity, _ = random.choice(list(selection.items()))

                # (optionally) cross random players
                if crossover_rate > random.uniform(0, 1) and epoch > 0:

                    rnd_recessive, _ = random.choice(
                        list(selection.items())
                    )

                    rnd_entity = cross(rnd_entity, rnd_recessive)

                # mutate random selected
                mut_entity = mutate(rnd_entity, mutation_rate)

                # calculate score
                population[mut_entity] = mut_entity.accuracy(batch)

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
                "[--- @{:02}: \t avg(train)={:2.4f} \t best(train)={:2.4f} \t best(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    (epoch + 1),
                    sum(population.values()) / len(population),
                    score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )
