from geneticNLP.neural import amoeba

from geneticNLP.utils import time_track
from geneticNLP.tasks.utils import setup, evaluate

#
#
#  -------- do_amoeba -----------
#
@time_track
def do_amoeba(args: dict) -> None:
    print("\n[--- AMOEBA ---]")

    # --- setup experiment
    model, data, utils = setup(args)

    # --- start amoeba
    model = amoeba(
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
