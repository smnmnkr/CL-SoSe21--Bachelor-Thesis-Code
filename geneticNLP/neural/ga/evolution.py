import random, operator

from datetime import datetime

from torch.utils.data import IterableDataset

from geneticNLP.neural.ga.mutation import mutate
from geneticNLP.neural.ga.selection import elitism

from geneticNLP.data import batch_loader

from geneticNLP.utils import get_device


def evolve(
    Model_cls: object,
    config: dict,
    encoding: object,
    embedding: object,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    mutation_rate: float = 0.2,
    population_size: int = 50,
    survivor_rate: float = 4,
    epoch_num: int = 60,
    report_rate: int = 10,
    batch_size: int = 16,
):

    # create batched loader
    train_loader = batch_loader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
    )
    dev_loader = batch_loader(
        dev_set,
        batch_size=batch_size,
        num_workers=0,
    )

    # generate base population
    population: dict = {
        Model_cls(config).to(get_device()): 0.0
        for _ in range(population_size)
    }

    # --
    for t in range(epoch_num):
        time_begin = datetime.now()

        # -- evaluate score on dev set
        for modul, _ in population.items():

            population[modul] = modul.evaluate(train_loader)

        # --- report
        if (t + 1) % report_rate == 0:
            best, score = max(
                population.items(), key=operator.itemgetter(1)
            )

            print(
                "@{:02}: \t acc(train)={:2.4f} \t acc(dev)={:2.4f} \t time(epoch)={}".format(
                    (t + 1),
                    score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

        # --- selection
        selection: dict = elitism(population, survivor_rate)
        next_generation: list = []

        # --- mutation
        for _ in range(population_size):

            # select random player from selection
            random_selected, _ = random.choice(list(selection.items()))

            # mutate random selected
            random_mutated = mutate(random_selected, mutation_rate)

            # mutate and append it
            next_generation.append(random_mutated)

        population = dict.fromkeys(next_generation, 0.0)
