from geneticNLP.neural import swarm
from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import setup, evaluate


#
#
#  -------- do_swarm -----------
#
@time_track
def do_swarm(args: dict) -> None:

    # --- load config json files
    model_config: dict = load_json(args.model_config)
    data_config: dict = load_json(args.data_config)
    swarm_config: dict = load_json(args.swarm_config)

    # --- setup experiment
    model, data, utils = setup(model_config, data_config)

    # --- start hybrid
    model = swarm(
        utils.get("model_class"),
        utils.get("model_config"),
        data.get("train"),
        data.get("dev"),
        **swarm_config,
    )

    # --- run metric
    evaluate(
        model,
        utils.get("encoding"),
        data.get("test"),
    )
