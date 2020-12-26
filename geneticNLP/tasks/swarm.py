from geneticNLP.neural import swarm

from geneticNLP.utils import time_track
from geneticNLP.tasks.utils import (
    setup,
    init_population,
    evaluate,
)


#
#
#  -------- do_swarm -----------
#
@time_track
def do_swarm(args: dict) -> None:
    print("\n[--- SWARM OPTIMIZATION ---]")

    # --- setup experiment
    _, data, utils = setup(args)

    # --- init population
    population = init_population(
        utils.get("model_class"),
        utils.get("model_config"),
        utils.get("train_config").get("population_size"),
    )

    # --- start hybrid
    model = swarm(
        population,
        data.get("train"),
        data.get("dev"),
        **utils.get("train_config").get("parameters"),
    )

    # --- run metric
    evaluate(
        model,
        utils.get("encoding"),
        data.get("test"),
    )
