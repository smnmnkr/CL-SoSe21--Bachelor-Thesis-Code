from geneticNLP.models import POSTagger

from geneticNLP.neural import train

from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import load_resources


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

    # --- load external data sources
    embedding, encoding, data = load_resources(data_config)

    # --- add data dependent model config
    model_config["lstm"]["input_size"] = embedding.dimension
    model_config["score"]["output_size"] = len(encoding)

    # --- create model
    model = POSTagger(model_config)

    # --- start training
    train(
        model,
        data.get("train"),
        data.get("dev"),
        **training_config,
    )
