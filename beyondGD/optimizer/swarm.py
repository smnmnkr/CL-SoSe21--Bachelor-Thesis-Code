from datetime import datetime
from typing import Generator

import torch

from beyondGD.data import batch_loader

from beyondGD.optimizer.util import (
    get_normal_TT,
    get_rnd_prob,
    copy_model,
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
    initial_velocity_rate: float = 0.02,
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
        Particle(model, initial_velocity_rate)
        for (model, _) in population.items()
    ]

    # -- epoch loop
    for epoch in range(1, epoch_num + 1):
        time_begin: datetime = datetime.now()

        # -- batch loop
        for batch in train_loader:

            # get the best particle
            if epoch == 1:
                gbest: Particle = get_best(swarm, batch)

            # -- particle update loop
            for i in range(len(swarm)):

                if gbest == swarm[i]:
                    continue

                swarm[i].update_position(
                    gbest,
                    learning_rate=learning_rate,
                    velocity_weight=velocity_weight,
                    personal_weight=personal_weight,
                    global_weight=global_weight,
                )

            # -- particle evaluate loop
            for i in range(len(swarm)):
                if swarm[i].fitness(batch) >= swarm[i].fitness(
                    batch, best=True
                ):
                    swarm[i].update_best()

                    if swarm[i].fitness(batch) >= gbest.fitness(batch):
                        gbest = swarm[i]

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
                    gbest.network.evaluate(train_loader),
                    gbest.network.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    # --- return population
    return {
        particle.network: particle.network.evaluate(train_loader)
        for particle in swarm
    }


#  -------- get_best -----------
#
def get_best(swarm: list, batch: list) -> int:

    best: dict = None
    gscore: float = 0.0

    for particle in swarm:
        pscore: float = particle.fitness(batch)

        if pscore >= gscore:
            best = particle
            gscore = pscore

    return best


#
#
#  -------- Particle -----------
#
class Particle:

    network: Module
    best: Module
    velocity: list

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(
        self,
        model: Module,
        velocity: float,
    ) -> None:

        self.network = model
        self.best = copy_model(model)
        self.velocity = list(self.init_velocity(velocity))

    #
    #
    #  -------- init_velocity -----------
    #
    def init_velocity(
        self,
        velocity: float = 0.02,
    ) -> Generator:

        for param in self.network.parameters():
            yield (
                get_normal_TT(
                    param.shape,
                    velocity,
                )
            )

    #
    #
    #  -------- update_position -----------
    #
    def update_position(
        self,
        gbest: "Particle",
        learning_rate: float = 0.001,
        velocity_weight: float = 1.0,
        personal_weight: float = 1.0,
        global_weight: float = 1.0,
    ) -> None:

        updated_velocity: list = []

        for param, vel, p, g in zip(
            self.network.parameters(),
            self.velocity,
            self.best.parameters(),
            gbest.network.parameters(),
        ):

            updated_velocity.append(
                velocity_weight * vel
                + (personal_weight * get_rnd_prob() * (p.data - param.data))
                + (global_weight * get_rnd_prob() * (g.data - param.data))
            )

            param.data += learning_rate * updated_velocity[-1]

        self.velocity = updated_velocity

    #
    #
    #  -------- update_best -----------
    #
    def update_best(self) -> None:
        self.best = copy_model(self.network)

    #
    #
    #  -------- fitness -----------
    #
    def fitness(self, batch: list, best: bool = False) -> None:

        if not best:
            return self.network.accuracy(batch)

        else:
            return self.best.accuracy(batch)
