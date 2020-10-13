from geneticNLP.data import CONLLU
from geneticNLP.embeddings import FastText
from geneticNLP.models.postagger import POSTagger, batch_loss, accuracy

from geneticNLP.neural import train

from geneticNLP.utils import Encoding


data_train = CONLLU("./data/universal-dependencies--en-dev.conllu")
data_dev = CONLLU("./data/universal-dependencies--en-test.conllu")


model = POSTagger(
    {
        "lstm": {
            "hidden_size": 50,
            "depth": 2,
            "dropout": 0.5,
        },
        "score": {
            "hidden_size": 50,
            "dropout": 0.2,
        },
    },
    Encoding({tok.pos for sent in data_train for tok in sent}),
    FastText("./data/fasttext--cc.en.300.bin"),
)


train(
    model,
    data_train,
    data_dev,
    batch_loss,
    accuracy,
    report_rate=1,
)
