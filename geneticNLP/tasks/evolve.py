from geneticNLP.models import POSTagger

from geneticNLP.neural import evolve

from geneticNLP.utils import load_json, time_track
from geneticNLP.tasks.utils import load_resources

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

    # --- load external data sources
    embedding, encoding, data = load_resources(data_config)

    # --- add data dependent model config
    model_config["lstm"]["input_size"] = embedding.dimension
    model_config["score"]["output_size"] = len(encoding)

    # --- start evolution
    evolve(
        POSTagger,
        model_config,
        data.get("train"),
        data.get("dev"),
        **evolution_config,
    )
