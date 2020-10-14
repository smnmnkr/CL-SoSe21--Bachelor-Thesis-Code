import itertools

#  -------- elitism -----------
#
def elitism(generation: dict, n: int):

    ranking: dict = {
        k: v
        for k, v in sorted(generation.items(), key=lambda item: item[1])
    }

    return dict(itertools.islice(ranking.items(), len(ranking) - n, None))
