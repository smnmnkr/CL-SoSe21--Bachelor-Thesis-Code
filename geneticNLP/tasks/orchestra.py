from geneticNLP.neural import descent, evolve, swarm, amoeba

from geneticNLP.utils import time_track
from geneticNLP.tasks.utils import setup, evaluate

#
#
#  -------- do_orchestra -----------
#
@time_track
def do_orchestra(args: dict) -> None:
    print("\n[--- ORCHESTRA ---]")

    # --- setup experiment
    model, data, utils = setup(args)

    # --- start orchestra
    # TODO

    # --- run metric
    evaluate(
        model,
        utils.get("encoding"),
        data.get("test"),
    )
