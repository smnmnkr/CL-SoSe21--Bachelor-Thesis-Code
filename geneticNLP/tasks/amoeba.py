from geneticNLP.neural import amoeba

from geneticNLP.utils import time_track, dict_max
from geneticNLP.tasks.utils import (
    setup,
    init_population,
    evaluate,
)

#
#
#  -------- do_amoeba -----------
#
@time_track
def do_amoeba(args: dict) -> None:
    print("\n[--- AMOEBA ---]")

    # --- setup experiment
    _, data, utils = setup(args)

    # --- init population
    population = init_population(
        utils.get("model_class"),
        utils.get("model_config"),
        utils.get("train_config").get("population_size"),
    )

    # --- start amoeba
    amoeba(
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
