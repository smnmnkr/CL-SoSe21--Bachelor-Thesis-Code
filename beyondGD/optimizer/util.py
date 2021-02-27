import random
import copy

import multiprocessing as mp

import torch

from beyondGD.utils import get_device
from beyondGD.utils.type import TT, IterableDataset, Module


#
#
#  -------- evaluate_on_loader -----------
#  !!! mutable object population !!!
#
def evaluate_on_loader(
    population: dict,
    data_loader: IterableDataset,
) -> dict:

    return {
        entity: entity.evaluate(data_loader)
        for entity, _ in population.items()
    }


#
#
#  -------- accuracy_on_batch -----------
#
def accuracy_on_batch(
    population: dict,
    batch: list,
    mutliprocess: bool = False,
) -> dict:

    # calculate accuracy using pool multiprocess
    if mutliprocess:

        pool: mp.Pool = mp.Pool(mp.cpu_count())

        return_async: list = [
            (entity, pool.apply_async(entity.accuracy, args=(batch,)))
            for entity, _ in population.items()
        ]

        processed_population: dict = {
            row[0]: row[1].get() for row in return_async
        }

        pool.close()
        pool.join()

        return processed_population

    # calculate accuracy linear
    else:
        return {
            entity: entity.accuracy(batch)
            for entity, _ in population.items()
        }


#
#
#  -------- get_normal_TT -----------
#
def get_normal_TT(
    shape: tuple,
    variance: float,
) -> TT:
    return (
        torch.empty(shape)
        .normal_(
            mean=0,
            std=variance,
        )
        .to(get_device())
    )


#
#
#  -------- get_rnd_entity -----------
#
def get_rnd_entity(
    population: dict,
) -> Module:

    entity, _ = random.choice(list(population.items()))

    return entity


#
#
#  -------- get_rnd_prob -----------
#
def get_rnd_prob() -> float:
    return random.uniform(0, 1)


#
#
#  -------- copy_model -----------
#
def copy_model(
    model: Module,
) -> Module:
    return copy.deepcopy(model)


#
#
#  -------- copy_parameters -----------
#
def copy_parameters(
    model: Module,
) -> list:
    return copy.deepcopy(list(model.parameters()))
