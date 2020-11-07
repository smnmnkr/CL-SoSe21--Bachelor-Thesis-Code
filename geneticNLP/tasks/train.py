from geneticNLP.neural import train

from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import setup, evaluate


#
#
#  -------- do_train -----------
#
@time_track
def do_train(args: dict) -> None:

    # --- load config json files
    model_config: dict = load_json(args.model_config)
    training_config: dict = load_json(args.training_config)
    data_config: dict = load_json(args.data_config)

    # --- setup experiment
    model, data, encoding = setup(model_config, data_config)

    # --- start training
    train(
        model,
        data.get("train"),
        data.get("dev"),
        **training_config,
    )

    # --- run metric
    evaluate(
        model,
        encoding,
        data.get("test"),
    )
