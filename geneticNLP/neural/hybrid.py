import random

from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from geneticNLP.data import batch_loader
from geneticNLP.utils.methods import get_device


#
#
#  -------- hybrid -----------
#
def hybrid(
    model: nn.Module,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    noise_std: float = 0.1,
    learning_rate: float = 0.001,
    convergence_min: int = 0.95,
    population_size: int = 200,
    report_rate: int = 2,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # load dev set as batched loader
    dev_loader = batch_loader(
        dev_set,
        batch_size=batch_size,
        num_workers=0,
    )

    # start convergence, epoch
    convergence: float = 0.0
    epoch: int = 0

    # generate queen, swarm
    queen: nn.Module = model
    swarm: dict = {model: 0.0 for _ in range(population_size)}

    # --
    while convergence < convergence_min:
        time_begin = datetime.now()

        # load train set as batched loader
        # TODO: enable adaptive batch size
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
            num_workers=0,
        )

        for batch in train_loader:

            # --- init entitiy randomly and calculate batch score
            for entitiy, _ in swarm.items():
                swarm[entitiy] = entitiy.accuracy(batch)

            # --- update queen model
            scores = torch.tensor(list(swarm.values())).to(get_device())

            scores_std = (scores - torch.mean(scores)) / torch.std(scores)

            swarm_params: list = [
                [param for param in worker.parameters()] for worker in swarm
            ]

            for id_p, q_param in enumerate(queen.parameters()):

                for id_s, s_param in enumerate(
                    [worker[id_p] for worker in swarm_params]
                ):
                    q_param.data += (
                        learning_rate
                        / (population_size * noise_std)
                        * (s_param * scores_std[id_s])
                    )

        # --- increase epoch
        epoch += 1

        # --- report
        if (epoch + 1) % report_rate == 0:
            print(
                "[--- @{:02}: \t swarm(train)={:2.4f} \t queen(train)={:2.4f} \t queen(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    (epoch + 1),
                    sum(swarm.values()) / len(swarm),
                    queen.evaluate(train_loader),
                    queen.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )
