import random
import copy

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
) -> None:

    for entity, _ in population.items():
        population[entity] = entity.evaluate(data_loader)

    return None


#
#
#  -------- accuracy_on_batch -----------
#  !!! mutable object population !!!
#
def accuracy_on_batch(
    population: dict,
    batch: list,
) -> None:

    for entity, _ in population.items():
        population[entity] = entity.accuracy(batch)

    return None


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
