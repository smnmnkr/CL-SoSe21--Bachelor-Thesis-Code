from datetime import datetime
import itertools

import torch

from beyondGD.data import batch_loader
from beyondGD.utils import dict_max

from beyondGD.optimizer.util import (
    evaluate_on_loader,
    accuracy_on_batch,
    get_rnd_entity,
    get_rnd_prob,
    get_normal_TT,
    copy_model,
)

from beyondGD.utils import get_device
from beyondGD.utils.type import IterableDataset, Module


#
#
#  -------- evolve -----------
#
def evolve(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    mutation_rate: float = 0.02,
    selection_rate: int = 10,
    crossover_rate: float = 0.5,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # save population size
    population_size: int = len(population)

    # load train set as batched loader
    train_loader = batch_loader(
        train_set,
        batch_size=batch_size,
    )

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        for batch in train_loader:

            # --- calculate accuracy on batch
            population = accuracy_on_batch(population, batch)

            # --- select by elite
            selection: dict = elitism(population, selection_rate)

            # delete old population
            population.clear()

            # --- fill new population with mutated, crossed entities
            for _ in range(population_size):

                # get random players from selection
                entity = get_rnd_entity(selection)

                # (optionally) cross random players
                if crossover_rate > get_rnd_prob():

                    entity = crossover(entity, get_rnd_entity(selection))

                # mutate random selected
                mut_entity = mutate(entity, mutation_rate)

                # add to next generation
                population[mut_entity] = 0.0

        # --- report
        if epoch % report_rate == 0:

            # --- evaluate all models on train set
            evaluate_on_loader(population, train_loader)

            # --- find best model and corresponding score
            best, train_score = dict_max(population)

            # load dev set as batched loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
                num_workers=0,
            )

            print(
                "[--- @{:02}: \t avg(train)={:2.4f} \t best(train)={:2.4f} \t best(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    sum(population.values()) / len(population),
                    train_score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    return population


#
#
#  -------- elitism -----------
#
def elitism(generation: dict, n: int):

    ranking: dict = {
        k: v
        for k, v in sorted(generation.items(), key=lambda item: item[1])
    }

    return dict(itertools.islice(ranking.items(), len(ranking) - n, None))


#
#
#  -------- mutate -----------
#
@torch.no_grad()
def mutate(
    parent_network: Module,
    mutation_rate: float = 0.02,
):

    child_network: Module = copy_model(parent_network)

    for param in child_network.parameters():

        param.data += get_normal_TT(
            param.shape,
            mutation_rate,
        )

    return child_network


#
#
#  -------- crossover -----------
#
@torch.no_grad()
def crossover(
    dom_network: Module,
    rec_network: Module,
    dominance: float = 0.5,
):

    child_network: Module = copy_model(dom_network)

    for c_layer, r_layer in zip(
        child_network.parameters(),
        rec_network.parameters(),
    ):

        # create mask, which determines the retained weights
        mask_positive = (
            (torch.FloatTensor(c_layer.shape).uniform_() > dominance)
            .int()
            .to(get_device())
        )

        # create inverted mask
        mask_negativ = torch.abs(mask_positive - 1).to(get_device())

        # combine the two layers
        c_layer.data = c_layer * mask_positive + r_layer * mask_negativ

    return child_network
