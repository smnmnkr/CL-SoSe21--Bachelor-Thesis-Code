from datetime import datetime

import torch
import copy
import random

from beyondGD.data import batch_loader
from beyondGD.utils import dict_max, get_device

from beyondGD.utils.types import IterableDataset


#
#
#  -------- swarm -----------
#
def swarm(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    learning_rate: float = 0.001,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # load train set as batched loader
    train_loader = batch_loader(
        train_set,
        batch_size=batch_size,
    )

    # -- initial setup
    particles: list = [
        {
            "id": id,
            "model": model,
            "velocity": list(create_velocity(model)),
            "score": model.evaluate(train_loader),
            "best_params": copy.deepcopy(
                list(model.parameters())
            ),
        }
        for id, (model, _) in enumerate(population.items())
    ]

    # get the best
    global_best = get_best(particles)

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        for batch in train_loader:

            # calculate score of each particle in population
            for part in particles:

                part = update_position(
                    part,
                    global_best,
                    learning_rate=learning_rate,
                )

                if (
                    part["model"].accuracy(batch)
                    > part["score"]
                ):

                    part["score"] = part["model"].accuracy(
                        batch
                    )

                    part["best_params"] = copy.deepcopy(
                        list(part["model"].parameters())
                    )

                    if part["score"] > global_best["score"]:
                        global_best = part

        # --- report
        if epoch % report_rate == 0:

            # load dev set as batched loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
            )

            print(
                "[--- @{:02}: \t acc(train)={:2.4f} \t acc(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    global_best["model"].evaluate(train_loader),
                    global_best["model"].evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    # --- return population
    return population


#
#
#  -------- create_velocity -----------
#
def create_velocity(
    network,
    mutation_rate: float = 0.02,
):

    for param in network.parameters():

        yield (
            torch.empty(param.shape)
            .normal_(
                mean=0,
                std=mutation_rate,
            )
            .to(get_device())
        )


#
#
#  -------- update_position -----------
#
def update_position(
    particle,
    global_best,
    learning_rate: float = 0.001,
    velocity_weight: float = 1.0,
    personal_weight: float = 1.0,
    global_weight: float = 1.0,
):

    updated_model = copy.deepcopy(particle["model"])

    for param, vel, par, glo in zip(
        updated_model.parameters(),
        particle["velocity"],
        particle["best_params"],
        global_best["best_params"],
    ):

        param.data += learning_rate * (
            velocity_weight * vel
            + (
                personal_weight
                * random.uniform(0, 1)
                * (par - param.data)
            )
            + (
                global_weight
                * random.uniform(0, 1)
                * (glo - param.data)
            )
        )

    particle["model"] = updated_model
    particle["model"] = updated_model
    return particle


#  -------- get_best -----------
#
def get_best(particles) -> int:

    best = None
    score: float = -1.0

    for p in particles:

        if p["score"] > score:
            score = p["score"]
            best = p

    return best