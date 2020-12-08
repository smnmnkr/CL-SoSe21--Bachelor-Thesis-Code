from geneticNLP.neural import swarm

from geneticNLP.utils import time_track
from geneticNLP.tasks.utils import setup, evaluate


#
#
#  -------- do_swarm -----------
#
@time_track
def do_swarm(args: dict) -> None:
    print("\n[--- SWARM OPTIMIZATION ---]")

    # --- setup experiment
    model, data, utils = setup(args)

    # --- start hybrid
    model = swarm(
        model,
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
