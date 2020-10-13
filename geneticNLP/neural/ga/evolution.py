import itertools
import copy
import random

from datetime import datetime

import torch
import torch.nn as nn

from torch.utils.data import IterableDataset

from geneticNLP.neural.ga.mutation import mutate
from geneticNLP.neural.ga.selection import elitism


def evolve(
    start_model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    accuracy: callable,
    mutation_rate: float = 1e-2,
    survivor_rate: float = 1e-6,
    population_size: int = 5,
    epoch_num: int = 60,
    report_rate: int = 10,
):

    # activate gpu usage for the model if possible, else nothing will change
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_model = start_model.to(device)

    # generate base population
    population = dict.fromkeys(
        list(itertools.repeat(copy.deepcopy(start_model), population_size)),
        0.0,
    )

    # --
    for t in range(epoch_num):
        time_begin = datetime.now()

        # -- eval
        for modul, score in population.items():
            modul.eval()
            with torch.no_grad():
                population[modul] = accuracy(modul, train_set)

        selection: dict = elitism(population, 5)
        next_generation: list = []

        # generate new players
        for _ in range(population_size):

            # select random player from selection
            randomSelected, _ = random.choice(list(selection.items()))

            # mutate and append it
            next_generation.append(mutate(randomSelected, mutation_rate))

        population = dict.fromkeys(next_generation, 0.0)

        # reporting (every `report_rate` epochs)
        if (t + 1) % report_rate == 0:

            best, _ = next(iter(population.items()))

            train_acc = accuracy(best, train_set)

            # get dev accurracy if given
            if dev_set:
                dev_acc = accuracy(best, dev_set)
            else:
                dev_acc = 0.0

            print(
                f"@{(t + 1):02}: \t acc(train)={train_acc:2.4f}  \t acc(dev)={dev_acc:2.4f}  \t time(epoch)={datetime.now() - time_begin}"
            )
