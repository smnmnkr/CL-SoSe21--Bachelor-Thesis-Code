from geneticNLP.neural import evolve

from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import setup, evaluate

#
#
#  -------- do_evolve -----------
#
@time_track
def do_evolve(args: dict) -> None:
    print("\n[--- EVOLUTION ---]")

    # --- load config json files
    model_config: dict = load_json(args.model_config)
    evolution_config: dict = load_json(args.evolution_config)
    data_config: dict = load_json(args.data_config)

    # --- setup experiment
    model, data, utils = setup(model_config, data_config)

    # --- start evolution
    model = evolve(
        utils.get("model_class"),
        utils.get("model_config"),
        data.get("train"),
        data.get("dev"),
        **evolution_config,
    )

    # --- run metric
    evaluate(
        model,
        utils.get("encoding"),
        data.get("test"),
    )
