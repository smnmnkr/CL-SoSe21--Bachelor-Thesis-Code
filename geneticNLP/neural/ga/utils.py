import random, multiprocessing

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

    # --- evaluate all models on train set
    for entity, _ in population.items():

        def worker_eval(entity):
            population[entity] = entity.evaluate(data_loader)

        processes = multiprocessing.Process(
            target=worker_eval, args=(entity,)
        )

        processes.start()

    return None


#
#
#  -------- process_non_parallel -----------
#  !!! mutable object population !!!
def process_non_parallel(
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
