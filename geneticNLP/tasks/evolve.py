from geneticNLP.neural import evolve

from geneticNLP.utils import time_track
from geneticNLP.tasks.utils import setup, evaluate

#
#
#  -------- do_evolve -----------
#
@time_track
def do_evolve(args: dict) -> None:
    print("\n[--- EVOLUTION ---]")

    # --- setup experiment
    model, data, utils = setup(args)

    # --- start evolution
    model = evolve(
        utils.get("model_class"),
        utils.get("model_config"),
        data.get("train"),
        data.get("dev"),
        **utils.get("train_config"),
    )

    # --- run metric
    evaluate(
        model,
        utils.get("encoding"),
        data.get("test"),
    )
