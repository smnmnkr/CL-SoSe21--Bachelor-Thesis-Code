from geneticNLP.data import PreProcessed
from geneticNLP.embeddings import FastText
from geneticNLP.models import POSTagger

from geneticNLP.neural import train, evolve

from geneticNLP.utils import Encoding


# 0 = evolve, 1 = train
mode: int = 0

ud_tags: set = {
    "ADV",
    "SCONJ",
    "ADP",
    "PRON",
    "PUNCT",
    "AUX",
    "NOUN",
    "PROPN",
    "INTJ",
    "CCONJ",
    "PART",
    "X",
    "NUM",
    "ADJ",
    "SYM",
    "DET",
    "VERB",
}


enc = Encoding(ud_tags)
emb = FastText("./data/fasttext--cc.en.300.bin")

data_train_processed = PreProcessed(
    "./data/universal-dependencies--en-dev_reduced.conllu", emb, enc
)

data_dev_processed = PreProcessed(
    "./data/universal-dependencies--en-test.conllu", emb, enc
)


config: dict = {
    "lstm": {
        "input_size": emb.dimension,
        "hidden_size": 16,
        "depth": 1,
        "dropout": 0.0,
    },
    "score": {
        "hidden_size": 16,
        "output_size": len(enc),
        "dropout": 0.0,
    },
}


if __name__ == "__main__":

    if mode == 0:
        evolve(
            POSTagger,
            config,
            data_train_processed,
            data_dev_processed,
        )

    elif mode == 1:
        train(
            POSTagger(config),
            data_train_processed,
            data_dev_processed,
        )
