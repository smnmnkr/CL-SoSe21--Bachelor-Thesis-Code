from geneticNLP.neural import evolve

from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import setup

#
#
#  -------- do_evolve -----------
#
@time_track
def do_evolve(args: dict) -> None:

    # --- load config json files
    model_config: dict = load_json(args.model_config)
    evolution_config: dict = load_json(args.evolution_config)
    data_config: dict = load_json(args.data_config)

    # --- setup experiment
    model, data = setup(model_config, data_config)

    # --- start evolution
    evolve(
        model,
        data.get("train"),
        data.get("dev"),
        **evolution_config,
    )
