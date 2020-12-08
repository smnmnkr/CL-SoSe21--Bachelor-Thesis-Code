from datetime import datetime

import torch

from geneticNLP.data import batch_loader
from geneticNLP.utils import dict_max, dict_min

from geneticNLP.neural.ga.utils import evaluate_linear

from geneticNLP.utils import get_device
from geneticNLP.utils.types import Module, IterableDataset


#
#
#  -------- amoeba -----------
#
def amoeba(
    model_CLS: Module,
    config: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    population_size: int = 80,
    epoch_num: int = 200,
    report_rate: int = 10,
    batch_size: int = 32,
):
    # disable gradients
    torch.set_grad_enabled(False)

    # generate set of arbitrary models
    X: dict = {
        model_CLS(config).to(get_device()): 0.0
        for _ in range(population_size)
    }

    # --
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        # load train set as batched loader
        train_loader = batch_loader(
            train_set,
            batch_size=batch_size,
        )

        for batch in train_loader:

            # calculate score of each entity
            for entity, _ in X.items():
                X[entity] = entity.accuracy(batch)

            # get worst point
            worst, score = dict_min(X)

            # calculate the center of mass of the remaining points

            # remove worst
            all(map(X.pop, {worst: score}))

            # get the mass of each object in X
            # TODO: consider normalizing the mass values
            X_mass = torch.tensor(list(X.values())).to(get_device())

            # get the parameters/location of each object in X
            X_params: list = [
                [param for param in entity.parameters()] for entity in X
            ]

            # optimize the parameters of the worst entity
            for w_id, w_param in enumerate(worst.parameters()):

                center_of_mass = torch.empty(w_param.shape).to(get_device())

                for id_s, s_param in enumerate(
                    [param[w_id] for param in X_params]
                ):
                    center_of_mass += (s_param * X_mass[id_s]) / X_mass[
                        id_s
                    ]

                w_param.data += center_of_mass

            # insert worst back into X as best, skip accuracy calc
            X[worst] = 0.0

        # --- report
        if epoch % report_rate == 0:

            # --- evaluate all models on train set
            evaluate_linear(X, train_loader)

            # --- find best model and corresponding score
            best, score = dict_max(X)

            # load dev set as batched loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
                num_workers=0,
            )

            print(
                "[--- @{:02}: \t avg(train)={:2.4f} \t best(train)={:2.4f} \t best(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    sum(X.values()) / len(X),
                    score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    # --- return best model
    model, _ = dict_max(X)
    return model
