from datetime import datetime

import torch
from torch.optim import Adam

from beyondGD.data import batch_loader
from beyondGD.utils import dict_max

from beyondGD.optimizer.evolution import elitism, mutate, crossover

from beyondGD.optimizer.util import (
    evaluate_on_loader,
    # accuracy_on_batch,
    get_rnd_entity,
    get_rnd_prob,
)

from beyondGD.utils.type import TT, IterableDataset, Module, DataLoader


#
#
#  -------- gadam -----------
#
def gadam(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-6,
    mutation_rate: float = 0.02,
    mutation_prob: float = 0.5,
    crossover_prob: float = 0.5,
    selection_size: int = 10,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # enable gradients
    torch.set_grad_enabled(True)

    # save population size
    population_size: int = len(population)

    # load train set as batched loader
    train_loader: DataLoader = batch_loader(
        train_set,
        batch_size=batch_size,
    )

    # load dev set as batched loader
    dev_loader: DataLoader = batch_loader(
        dev_set,
        batch_size=batch_size,
        num_workers=0,
    )

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin: datetime = datetime.now()

        for batch in train_loader:

            population: dict = evaluate_on_loader(population, dev_loader)

            # --- calculate accuracy on batch
            # population: dict = accuracy_on_batch(population, batch)

            # --- select by elite
            selection: dict = elitism(population, selection_size)

            # delete old population
            population.clear()

            # --- fill new population with mutated, crossed entities
            for _ in range(population_size):

                # get random players from selection, disable dropout
                entity: Module = get_rnd_entity(selection)
                entity.train(False)

                # (optionally) cross random players
                if crossover_prob > get_rnd_prob():
                    entity: Module = crossover(
                        entity, get_rnd_entity(selection)
                    )

                # (optionally) mutate random players
                if mutation_prob > get_rnd_prob():
                    entity: Module = mutate(entity, mutation_rate)

                # enable dropout
                entity.train(True)

                # choose Adam for optimization
                # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
                optimizer = Adam(
                    entity.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
                optimizer.zero_grad()

                # compute loss, backward
                loss: TT = entity.loss(batch)
                loss.backward()

                # optimize
                optimizer.step()

                # reduce memory usage by deleting loss after calculation
                # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
                del loss

                # add to next generation
                population[entity] = 0.0

        # --- report
        if epoch % report_rate == 0:

            # --- evaluate all models on train set
            population: dict = evaluate_on_loader(population, dev_loader)

            # --- find best model and corresponding score
            best, dev_score = dict_max(population)

            print(
                "[--- @{:02}: \t avg(train)={:2.4f} \t best(train)={:2.4f} \t best(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    sum(population.values()) / len(population),
                    best.evaluate(train_loader),
                    dev_score,
                    datetime.now() - time_begin,
                )
            )

    return population
