from datetime import datetime

import torch

from beyondGD.data import batch_loader

from beyondGD.optimizer.util import (
    evaluate_on_loader,
    accuracy_on_batch,
    copy_model,
)

from beyondGD.utils import (
    dict_max,
    dict_min,
    get_device,
    smooth_gradient,
)
from beyondGD.utils.type import IterableDataset, Module


#
#
#  -------- simplex -----------
#
def simplex(
    population: dict,
    train_set: IterableDataset,
    dev_set: IterableDataset,
    epoch_num: int = 200,
    expansion_rate: float = 2.0,
    contraction_rate: float = 0.5,
    shrink_rate: float = 0.02,
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

            # --- calculate accuracy on batch
            population = accuracy_on_batch(population, batch)

            # get the best, worst particle
            p_best, b_score = dict_max(population)
            p_worst, w_score = dict_min(population)

            # remove worst
            all(map(population.pop, {p_worst: w_score}))

            _, sec_w_score = dict_min(population)

            # calculate center of mass
            C: list = centroid(population)

            p_ref = reflect(p_worst, C)
            p_new = None

            #
            # --- Case Handling

            # Reflect Case: Reflection Particle performs better then the worst
            if b_score >= p_ref.accuracy(batch) > sec_w_score:
                p_new = p_ref

            # Expansion Case: Reflection Particle is the new best
            elif p_ref.accuracy(batch) > b_score:
                p_new = expansion(p_worst, p_ref, C, batch, expansion_rate)

            # Outside Contraction Case: Reflection Particle performs as the second worst
            elif sec_w_score >= p_ref.accuracy(batch) > w_score:
                p_new = outside_contraction(
                    p_worst, p_ref, C, batch, contraction_rate
                )

            # Inside Contraction Case: Reflection Particle performs worst
            elif w_score >= p_ref.accuracy(batch):
                p_new = inside_contraction(
                    p_worst, p_ref, C, batch, -contraction_rate
                )

            # Shrinkage Case: Inside Contracted Particle performs worst
            if p_new is None:
                population = shrinkage(population, p_best, shrink_rate)

            else:
                population[p_new] = 0.0

        #
        # --- Report
        if epoch % report_rate == 0:

            # --- evaluate all models on train set
            evaluate_on_loader(population, train_loader)

            # --- find best model and corresponding score
            best, score = dict_max(population)

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
    P_mass = torch.tensor(list(population.values())).to(get_device())

    # get the parameters $param$ as tensors (coordinates) in $P$
    P_params: list = [
        [param for param in entity.parameters()] for entity in population
    ]

    # calcuate the center of mass $R$ for every parameter $param$ in every particle in $P$
    for w_id, w_param in enumerate(next(iter(population)).parameters()):

        # where $R$ is the center of mass
        # calculate: (1/M) * sum_{i=1}^{n} m_i*r_i
        R = torch.empty(w_param.shape).to(get_device())

        # where M is the total mass
        # calculate: sum_{i=1}^{n} m
        M = torch.empty(w_param.shape).to(get_device())

        # where $n$ is the id of the particle,
        # and $r$ the values (coordinates) as a tensor
        for n, r in enumerate([param[w_id] for param in P_params]):

            # add the values $r$ with respect to the mass $m$, here P_mass[n]
            R += r * P_mass[n]

            # add to the total mass $M$
            M += P_mass[n]

        # calculate center of mass with respect to the total mass
        R *= 1 / M

        # add total mass to tensor
        C.append(R)

    return C


#  -------- reflect -----------
#
def reflect(p, C, reflection_param: float = 1):

    p_ref: Module = copy_model(p)

    for param_ref, param_c in zip(p_ref.parameters(), C):

        param_c_smoothed = smooth_gradient(param_c).data

        param_ref.data = param_c_smoothed + reflection_param * (
            param_c_smoothed - param_ref.data
        )

    return p_ref


#  -------- expansion -----------
#
def expansion(
    p_worst,
    p_ref,
    C,
    batch,
    expansion_rate: float = 2.0,
):

    p_exp = reflect(p_worst, C, expansion_rate)

    if p_exp.accuracy(batch) > p_ref.accuracy(batch):
        return p_exp

    else:
        return p_ref


#  -------- outside_contraction -----------
#
def outside_contraction(
    p_worst,
    p_ref,
    C,
    batch,
    outside_contraction_rate: float = 0.5,
):

    p_contract = reflect(p_worst, C, outside_contraction_rate)

    if p_contract.accuracy(batch) > p_ref.accuracy(batch):
        return p_contract

    else:
        return p_ref


#  -------- inside_contraction -----------
#
def inside_contraction(
    p_worst,
    p_ref,
    C,
    batch,
    inside_contraction_rate: float = -0.5,
):

    p_contract = reflect(p_worst, C, inside_contraction_rate)

    if p_contract.accuracy(batch) > p_ref.accuracy(batch):
        return p_contract

    else:
        return None


#  -------- shrinkage -----------
#
def shrinkage(
    population: dict,
    p_best,
    shrink_rate: float = 0.02,
) -> dict:

    new_population: dict = {p_best: 0.0}

    for p in population:

        p_new = reflect(p, p_best.parameters(), shrink_rate)

        new_population[p_new] = 0.0

    return new_population
