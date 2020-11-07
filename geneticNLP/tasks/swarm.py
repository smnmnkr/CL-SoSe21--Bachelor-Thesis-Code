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
    model, data, encoding = setup(model_config, data_config)

    # --- start hybrid
    swarm(
        model,
        data.get("train"),
        data.get("dev"),
        **swarm_config,
    )

    # --- run metric
    evaluate(
        model,
        encoding,
        data.get("test"),
    )
