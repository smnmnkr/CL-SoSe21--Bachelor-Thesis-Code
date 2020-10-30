import random, multiprocessing
from multiprocessing import Manager

import torch.multiprocessing as mp

from geneticNLP.neural.ga import mutate, cross, elitism


# prevent MAC OSX multiprocessing bug
# src: https://github.com/pytest-dev/pytest-flask/issues/104
multiprocessing.set_start_method("fork")


#
#
#  -------- evaluate_parallel -----------
#  !!! mutable object population !!!
#
def evaluate_parallel(population: dict, data_loader):

    #
    #  -------- worker_eval -----------
    def worker_eval(entity):
        population[entity] = entity.evaluate(data_loader)

    #
    # --- begin method:

    processes: list = []

    for entity, _ in population.items():

        p = mp.Process(target=worker_eval, args=(entity,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return None


#
#
#  -------- evaluate_linear -----------
#  !!! mutable object population !!!
#
def evaluate_linear(population: dict, data_loader):

    for entity, _ in population.items():
        population[entity] = entity.evaluate(data_loader)

    return None


#
#
#  -------- process_parallel -----------
#  !!! mutable object population !!!
def process_parallel(
    population: dict,
    batch,
    population_size: int,
    selection_rate: int,
    crossover_rate: int,
    max_processes_active: int = 24,
):
    #
    #  -------- worker_process -----------
    def worker_process(_):
        # get random players from selection
        rnd_entity, score = random.choice(list(selection.items()))

        # (optionally) cross random players
        if crossover_rate > random.uniform(0, 1) and len(selection) > 1:

            rnd_recessive, _ = random.choice(list(selection.items()))

            rnd_entity = cross(rnd_entity, rnd_recessive)

        # mutate random selected
        mut_entity = mutate(rnd_entity, 1 - score)

        # calculate score
        new_population[mut_entity] = mut_entity.accuracy(batch)

    #
    # --- begin method:

    # create empty new generation
    new_population: dict = Manager().dict()

    # --- select by elite if is not first epoch else use only input model
    selection: dict = (
        elitism(population, selection_rate)
        if (len(population) > 1)
        else population
    )

    processes: list = []
    processes_active: int = 1

    # --- fill new population with mutated, crossed entities
    for n in range(population_size):

        p = mp.Process(target=worker_process, args=(n,))
        p.start()
        processes.append(p)

        processes_active += 1

        # FIXME: limit maximum number of parallele subprocesses
        # src: https://stackoverflow.com/questions/60049527/shutting-down-manager-error-attributeerror-forkawarelocal-object-has-no-attr
        if processes_active >= max_processes_active:
            while processes_active >= max_processes_active:
                processes_active = 0
                for p in processes:
                    processes_active += p.is_alive()

    for p in processes:
        p.join()

    return new_population


#
#
#  -------- process_linear -----------
#  !!! mutable object population !!!
def process_linear(
    population: dict,
    batch,
    population_size: int,
    selection_rate: int,
    crossover_rate: int,
):
    # create empty new generation
    new_population: dict = {}

    # --- select by elite if is not first epoch else use only input model
    selection: dict = (
        elitism(population, selection_rate)
        if (len(population) > 1)
        else population
    )

    # --- fill new population with mutated, crossed entities
    for _ in range(population_size):

        # get random players from selection
        rnd_entity, score = random.choice(list(selection.items()))

        # (optionally) cross random players
        if crossover_rate > random.uniform(0, 1) and len(selection) > 1:

            rnd_recessive, _ = random.choice(list(selection.items()))

            rnd_entity = cross(rnd_entity, rnd_recessive)

        # mutate random selected
        mut_entity = mutate(rnd_entity, 1 - score)

        # calculate score
        new_population[mut_entity] = mut_entity.accuracy(batch)

    return new_population
