from geneticNLP.models import Model
from geneticNLP.models.postagger import POSstripped, POSfull

from geneticNLP.encoding import Encoding
from geneticNLP.embeddings import FastText

from geneticNLP.data import PreProcessed, CONLLU
from geneticNLP.utils import time_track, get_device


#
#
#  -------- setup -----------
#
@time_track
def setup(
    model_config: dict,
    data_config: dict,
):

    # --- load external data sources
    embedding, encoding, data = load_resources(data_config, model_config)

    # --- load model
    model, model_config = load_tagger(
        model_config, data_config, embedding, encoding
    )

    return (model, data, encoding)


#
#
#  -------- load_tagger -----------
#
def load_tagger(
    model_config: dict,
    data_config: dict,
    embedding: FastText,
    encoding: Encoding,
) -> Model:

    # --- add data dependent model config
    model_config["lstm"]["in_size"] = embedding.dimension
    model_config["score"]["hid_size"] = len(encoding)

    # --- load stripped model
    if data_config.get("preprocess"):
        model = POSstripped(model_config).to(get_device())

    # --- load full model
    else:
        model = POSfull(model_config, embedding, encoding).to(get_device())

    # --- return model and updated config
    return (model, model_config)


#
#
#  -------- load_resources -----------
#
def load_resources(
    data_config: dict,
    model_config: dict,
) -> tuple:

    # --- try loading external resources
    try:
        # --- create embedding and encoding objects
        embedding = FastText(
            data_config.get("embedding"),
            dimension=model_config.get("embedding")["size"],
        )
        encoding = Encoding(data_config.get("encoding"))

        # --- load and preprocess train and dev data
        if data_config.get("preprocess"):
            data_train = PreProcessed(
                data_config.get("train"), embedding, encoding
            )
            data_dev = PreProcessed(
                data_config.get("dev"), embedding, encoding
            )

        # --- load train and dev data
        else:
            data_train = CONLLU(data_config.get("train"))
            data_dev = CONLLU(data_config.get("dev"))

    # --- handle file not found
    except FileNotFoundError as error:
        raise error

    # --- (optional) test data
    data_test: PreProcessed = None
    if data_config.get("test", None) != None:

        # --- load and preprocess test data
        if data_config.get("preprocess"):
            data_test = PreProcessed(
                data_config.get("test"), embedding, encoding
            )

        # --- test data
        else:
            data_test = CONLLU(data_config.get("test"))

    # --- return (FastText, Encoding, {"data": type})
    return (
        embedding,
        encoding,
        {"train": data_train, "dev": data_dev, "test": data_test},
    )
