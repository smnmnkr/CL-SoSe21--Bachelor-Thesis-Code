from datetime import datetime

import torch

from beyondGD.data import batch_loader
from beyondGD.utils import dict_max, dict_min

from beyondGD.neural.ga import mutate
from beyondGD.neural.ga.utils import evaluate_linear

from beyondGD.utils import get_device, smooth_gradient
from beyondGD.utils.types import IterableDataset


#
#
#  -------- amoeba -----------
#
def amoeba(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
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

    # -- train loop
    for epoch in range(1, epoch_num + 1):
        time_begin = datetime.now()

        # -- batch loop
        for batch in train_loader:

            if len(population) == 1:
                return population


            # calculate score of each particle in population
            for particle, _ in population.items():
                population[particle] = particle.accuracy(batch)

            # get the best, worst particle
            p_best, b_score = dict_max(population)
            p_worst, w_score = dict_min(population)

            # remove worst
            all(map(population.pop, {p_worst: w_score}))

            # calculate center of mass :
            C: list = centroid(population)

            p_ref = p_worst.copy(p_worst)
            p_new = p_worst.copy(p_worst)

            for ref_param, c_param in zip(
                p_ref.parameters(), C
            ):
                ref_param.data = (
                    2 * smooth_gradient(c_param)
                    - ref_param.data
                )

            # Expansion Case: Reflection Particle is the new best
            if p_ref.accuracy(batch) > b_score:
                p_new = expansion(p_ref, C, batch)

            # Median Case: Reflection Particle performs better then the worst
            elif b_score >= p_ref.accuracy(batch) > w_score:
                p_new = p_ref

            # Contraction Case: Reflection Particle performs the worst
            elif w_score > p_ref.accuracy(batch):
                contraction(p_ref, C)

            population[p_new] = 0.0

            # Shrinkage Case:
            if p_new.accuracy(batch) <= w_score:
                shrinkage(population, p_new, p_best)

        # --- report
        if epoch % report_rate == 0:

            # --- evaluate all models on train set
            evaluate_linear(population, train_loader)

            # --- find best model and corresponding score
            best, score = dict_max(population)

            # load dev set as batched loader
            dev_loader = batch_loader(
                dev_set,
                batch_size=batch_size,
                num_workers=0,
            )

            print(
                "[--- @{:02}: \t size={:02} \t avg(train)={:2.4f} \t best(train)={:2.4f} \t best(dev)={:2.4f} \t time(epoch)={} ---]".format(
                    epoch,
                    len(population),
                    sum(population.values()) / len(population),
                    score,
                    best.evaluate(dev_loader),
                    datetime.now() - time_begin,
                )
            )

    # --- return population
    return population


#  -------- centroid -----------
#
def centroid(population: dict) -> list:

    C: list = []

    # get the score (mass) of each model in $P$
    P_mass = torch.tensor(list(population.values())).to(
        get_device()
    )

    # get the parameters $param$ as tensors (coordinates) in $P$
    P_params: list = [
        [param for param in entity.parameters()]
        for entity in population
    ]

    # calcuate the center of mass $R$ for every parameter $param$ in every particle in $P$
    for w_id, w_param in enumerate(
        next(iter(population)).parameters()
    ):

        # where $R$ is the center of mass
        # calculate: (1/M) * sum_{i=1}^{n} m_i*r_i
        R = torch.empty(w_param.shape).to(get_device())

        # where M is the total mass
        # calculate: sum_{i=1}^{n} m
        M = torch.empty(w_param.shape).to(get_device())

        # where $n$ is the id of the particle,
        # and $r$ the values (coordinates) as a tensor
        for n, r in enumerate(
            [param[w_id] for param in P_params]
        ):

            # add the values $r$ with respect to the mass $m$, here P_mass[n]
            R += r * P_mass[n]

            # add to the total mass $M$
            M += P_mass[n]

        # calculate center of mass with respect to the total mass
        R *= 1 / M

        # add total mass to tensor
        C.append(R)

    return C


#  -------- expansion -----------
#
def expansion(p_ref, C, batch) -> None:

    p_exp = p_ref.copy(p_ref)

    for param_ref, param_c in zip(p_exp.parameters(), C):
        param_ref.data = (
            2 * param_ref.data - smooth_gradient(param_c).data
        )

    if p_exp.accuracy(batch) > p_ref.accuracy(batch):
        return p_exp

    else:
        return p_ref


#  -------- shrinkage -----------
#
def contraction(p_ref, C) -> None:
    for param_ref, param_c in zip(p_ref.parameters(), C):
        param_ref.data = 2 ** -1 * (
            param_ref.data - smooth_gradient(param_c).data
        )


#  -------- shrinkage -----------
#
def shrinkage(
    population: dict,
    p_new,
    p_best,
    mutation_rate: float = 0.5,
    mode: int = 1,
) -> None:

    # remove worst
    all(map(population.pop, {p_new: 0.0}))

    # custom approach:
    if mode == 0:

        len_pop = len(population)

        population.clear()

        population[p_best] = 0.0

        for _ in range(len_pop - 1):

            p_i, _ = mutate(p_best, mutation_rate)
            population[p_i] = 0.0

    # original approach:
    elif mode == 1:
        for p in population:
            for param_p, param_best in zip(
                p.parameters(),
                p_best.parameters(),
            ):
                param_p.data = 2 ** -1 * (
                    param_p.data + param_best.data
                )
