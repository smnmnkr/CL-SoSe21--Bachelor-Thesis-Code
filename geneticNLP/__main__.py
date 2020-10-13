from geneticNLP.data import CONLLU
from geneticNLP.embeddings import FastText, Untrained
from geneticNLP.models.postagger import POSTagger, batch_loss, accuracy

from geneticNLP.neural import train, evolve

from geneticNLP.utils import Encoding


data_train = CONLLU("./data/universal-dependencies--en-dev.conllu")
data_dev = CONLLU("./data/universal-dependencies--en-test.conllu")


model = POSTagger(
    {
        "lstm": {
            "hidden_size": 24,
            "depth": 2,
            "dropout": 0.5,
        },
        "score": {
            "hidden_size": 24,
            "dropout": 0.5,
        },
    },
    Encoding({tok.pos for sent in data_train for tok in sent}),
    Untrained({tok.word for sent in data_train for tok in sent}, 36),
    # FastText("./data/fasttext--cc.en.300.bin"),
)


evolve(
    model,
    data_train,
    data_dev,
    accuracy,
    epoch_num=6,
    report_rate=2,
)

# train(
#     model,
#     data_train,
#     data_dev,
#     batch_loss,
#     accuracy,
#     report_rate=10,
# )
