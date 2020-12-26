from geneticNLP.neural import evolve

from geneticNLP.utils import time_track, dict_max
from geneticNLP.tasks.utils import (
    setup,
    init_population,
    evaluate,
)

#
#
#  -------- do_evolve -----------
#
@time_track
def do_evolve(args: dict) -> None:
    print("\n[--- EVOLUTION ---]")

    # --- setup experiment
    _, data, utils = setup(args)

    # --- init population
    population = init_population(
        utils.get("model_class"),
        utils.get("model_config"),
        utils.get("train_config").get("population_size"),
    )

    # --- start evolution
    evolve(
        population,
        data.get("train"),
        data.get("dev"),
        **utils.get("train_config").get("parameters"),
    )

    best, _ = dict_max(population)

    # --- run metric
    evaluate(
        best,
        utils.get("encoding"),
        data.get("test"),
    )
