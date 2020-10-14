from geneticNLP.data import CONLLU, PreProcessed
from geneticNLP.embeddings import FastText
from geneticNLP.models import POSTagger

from geneticNLP.neural import train, evolve

from geneticNLP.utils import Encoding


data_train = CONLLU("./data/universal-dependencies--en-dev.conllu")
data_dev = CONLLU("./data/universal-dependencies--en-test.conllu")

enc = Encoding({tok.pos for sent in data_train for tok in sent})
emb = FastText("./data/fasttext--cc.en.300.bin")
# emb = Untrained({tok.word for sent in data_train for tok in sent}, 16),

data_train_processed = PreProcessed(
    "./data/universal-dependencies--en-dev.conllu", emb, enc
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


mode = 0

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
