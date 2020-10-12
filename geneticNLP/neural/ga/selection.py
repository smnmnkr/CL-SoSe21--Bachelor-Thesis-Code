#  -------- elitism -----------
#
def elitism(network_generation: list, n: int):

    ranking: list = sorted(
        network_generation, key=lambda network: network.getScore()
    )

    return ranking[len(ranking) - n :]
