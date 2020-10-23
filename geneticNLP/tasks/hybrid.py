from geneticNLP.neural import train, evolve

from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import setup


#
#
#  -------- do_hybrid -----------
#
@time_track
def do_hybrid(args: dict) -> None:

    # --- load config json files
    model_config: dict = load_json(args.model_config)
    training_config: dict = load_json(args.training_config)
    evolution_config: dict = load_json(args.evolution_config)
    data_config: dict = load_json(args.data_config)

    # --- setup experiment
    model, data = setup(model_config, data_config)

    # --- start training
    train(
        model,
        data.get("train"),
        data.get("dev"),
        **training_config,
    )

    # --- start evolution
    evolve(
        model,
        data.get("train"),
        data.get("dev"),
        **evolution_config,
    )
