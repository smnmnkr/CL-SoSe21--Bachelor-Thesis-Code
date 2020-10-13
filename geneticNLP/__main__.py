from geneticNLP.data import CONLLU
from geneticNLP.embeddings import FastText, Untrained
from geneticNLP.models import POSTagger

from geneticNLP.neural import train, evolve

from geneticNLP.utils import Encoding


data_train = CONLLU("./data/universal-dependencies--en-dev.conllu")
data_dev = CONLLU("./data/universal-dependencies--en-test.conllu")

enc = Encoding({tok.pos for sent in data_train for tok in sent})
emb = FastText("./data/fasttext--cc.en.300.bin")
# emb = Untrained({tok.word for sent in data_train for tok in sent}, 16),


model = POSTagger(
    {
        "lstm": {
            "input_size": emb.dimension,
            "hidden_size": 16,
            "depth": 1,
            "dropout": 0.0,
        },
        "score": {
            "hidden_size": 8,
            "output_size": len(enc),
            "dropout": 0.5,
        },
    },
)


# evolve(
#     model,
#     data_train,
#     data_dev,
#     accuracy,
#     epoch_num=6,
#     report_rate=2,
# )

train(
    model,
    enc,
    emb,
    data_train,
    data_dev,
    report_rate=10,
)
