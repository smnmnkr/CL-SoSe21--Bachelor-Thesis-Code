from geneticNLP.models import Model
from geneticNLP.models.postagger import POSstripped, POSfull

from geneticNLP.encoding import Encoding
from geneticNLP.embeddings import FastText

from geneticNLP.data import PreProcessed, CONLLU, batch_loader
from geneticNLP.utils import load_json, time_track, get_device


#
#
#  -------- setup -----------
#
@time_track
def setup(args: dict):

    # load config files from json
    model_config, train_config, data_config = load_config(args)

    # --- load external data sources
    embedding, encoding, data = load_resources(data_config, model_config)

    # --- load model
    model, CLS, model_config = load_tagger(
        model_config, data_config, embedding, encoding
    )

    return (
        model,
        data,
        # -- optional utils
        {
            "encoding": encoding,
            "embedding": embedding,
            "model_class": CLS,
            "model_config": model_config,
            "train_config": train_config,
            "data_config": data_config,
        },
    )


#
#
#  -------- load_config -----------
#
def load_config(args: dict) -> tuple:

    # --- load config json files
    try:
        model_config: dict = load_json(args.model_config)
        training_config: dict = load_json(args.training_config)
        data_config: dict = load_json(args.data_config)

    # TODO: handle error while loading
    except:
        raise Exception

    return (model_config, training_config, data_config)


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

    # --- set, load stripped model
    if data_config.get("preprocess"):
        CLS: Model = POSstripped
        model = CLS(model_config).to(get_device())

    # --- set, load full model
    else:
        CLS: Model = POSfull
        model = CLS(model_config, embedding, encoding).to(get_device())

    # --- return model and updated config
    return (model, CLS, model_config)


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
        # --- get POS-Tags from train and dev set
        taglist = CONLLU(data_config.get("train")).taglist.union(
            CONLLU(data_config.get("dev")).taglist
        )

        # --- create embedding and encoding objects
        embedding = FastText(
            data_config.get("embedding"),
            dimension=model_config.get("embedding")["size"],
        )
        encoding = Encoding(taglist)

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
        {
            "train": data_train,
            "dev": data_dev,
            "test": data_test,
        },
    )


#
#
#  -------- init_population -----------
#
def init_population(
    model_CLS: object,
    config: dict,
    size: int,
) -> dict:
    return {model_CLS(config).to(get_device()): 0.0 for _ in range(size)}


#
#
#  -------- evaluate -----------
#
def evaluate(
    model: Model,
    encoding: Encoding,
    data_set,
) -> None:

    print("[--- EVALUATION ---]")

    test_loader = batch_loader(data_set)

    model.evaluate(test_loader)
    model.metric.show(encoding=encoding)
