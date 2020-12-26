from geneticNLP.neural import descent

from geneticNLP.utils import time_track
from geneticNLP.tasks.utils import setup, evaluate


#
#
#  -------- do_train -----------
#
@time_track
def do_descent(args: dict) -> None:
    print("\n[--- GRADIENT DESCENT ---]")

    # --- setup experiment
    model, data, utils = setup(args)

    # --- start training
    descent(
        model,
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
