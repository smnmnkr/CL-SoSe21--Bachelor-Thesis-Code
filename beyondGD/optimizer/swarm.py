from datetime import datetime
from typing import Generator

import torch

from beyondGD.data import batch_loader

from beyondGD.optimizer.util import (
    get_normal_TT,
    get_rnd_prob,
    copy_model,
    copy_parameters,
)

from beyondGD.utils.type import IterableDataset, Module, DataLoader


#
#
#  -------- swarm -----------
#
def swarm(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    learning_rate: float = 0.001,
    velocity_weight: float = 1.0,
    personal_weight: float = 1.0,
    global_weight: float = 1.0,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # load train set as batched loader
    train_loader: DataLoader = batch_loader(
        train_set,
        batch_size=batch_size,
    )

    # -- initial swarm setup
    swarm: list = [
        {
            "id": id,
            "model": model,
            "velocity": list(create_velocity(model)),
            "best_score": model.evaluate(train_loader),
            "best_params": copy_parameters(model),
        }
        for id, (model, _) in enumerate(population.items())
    ]

    # save the best particle
    global_best: dict = get_best(swarm)

    # -- epoch loop
    for epoch in range(1, epoch_num + 1):
        time_begin: datetime = datetime.now()

        # -- batch loop
        for batch in train_loader:

            # -- particle loop
            for particle in swarm:

                particle: dict = update_position(
                    particle,
                    global_best,
                    learning_rate=learning_rate,
                    velocity_weight=velocity_weight,
                    personal_weight=personal_weight,
                    global_weight=global_weight,
                )

                if (
                    particle["model"].accuracy(batch)
                    > particle["best_score"]
                ):

                    particle["best_score"] = particle["model"].accuracy(
                        batch
                    )

                    particle["best_params"] = copy_parameters(
                        particle["model"]
                    )

                    if particle["best_score"] > global_best["best_score"]:
                        global_best = particle

        # --- report
        if epoch % report_rate == 0:

            # load dev set as batched loader
            dev_loader: DataLoader = batch_loader(
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
    return {
        particle["model"]: particle["model"].evaluate(train_loader)
        for particle in swarm
    }


#
#
#  -------- create_velocity -----------
#
def create_velocity(
    network: Module,
    boundary: float = 0.05,
) -> Generator:

    for param in network.parameters():
        yield (
            get_normal_TT(
                param.shape,
                boundary,
            )
        )


#
#
#  -------- update_position -----------
#
def update_position(
    particle: dict,
    global_best: dict,
    learning_rate: float = 0.001,
    velocity_weight: float = 1.0,
    personal_weight: float = 1.0,
    global_weight: float = 1.0,
) -> dict:

    updated_model: Module = copy_model(particle["model"])
    updated_vecolity: list = []

    for param, velo, pers, glob in zip(
        updated_model.parameters(),
        particle["velocity"],
        particle["best_params"],
        global_best["best_params"],
    ):

        updated_vecolity.append(
            velocity_weight * velo
            + (personal_weight * get_rnd_prob() * (pers - param.data))
            + (global_weight * get_rnd_prob() * (glob - param.data))
        )

        param.data += learning_rate * updated_vecolity[-1]

    particle["velocity"] = updated_vecolity
    particle["model"] = updated_model
    return particle


#  -------- get_best -----------
#
def get_best(swarm: list) -> int:

    best: dict = None
    score: float = -1.0

    for particle in swarm:

        if particle["best_score"] > score:
            score = particle["best_score"]
            best = particle

    return best
