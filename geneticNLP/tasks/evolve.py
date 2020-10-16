from geneticNLP.data import PreProcessed
from geneticNLP.embeddings import FastText
from geneticNLP.models import POSTagger

from geneticNLP.neural import evolve

from geneticNLP.utils import Encoding, load_json


#
#
#  -------- do_evolve -----------
#
def do_evolve(args: dict) -> None:

    # ---
    model_config: dict = load_json(args.model_config)
    evolution_config: dict = load_json(args.evolution_config)
    data_config: dict = load_json(args.data_config)

    try:
        # ---
        embedding = FastText(data_config.get("embedding"))
        encoding = Encoding(data_config.get("encoding"))

        # ---
        data_train = PreProcessed(
            data_config.get("train"), embedding, encoding
        )
        data_dev = PreProcessed(data_config.get("dev"), embedding, encoding)

    # ---
    except:
        raise NotImplementedError

    # ---
    model_config["lstm"]["input_size"] = embedding.dimension
    model_config["score"]["output_size"] = len(encoding)

    # ---
    evolve(
        POSTagger,
        model_config,
        data_train,
        data_dev,
        **evolution_config,
    )
