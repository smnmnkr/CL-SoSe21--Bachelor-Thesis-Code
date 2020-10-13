from geneticNLP.data import CONLLU

# NOTE: user dependent file
conllu_file: str = "./data/universal-dependencies--en-dev.conllu"

data_quantity: int = 2002


def test_loader():

    # create embedding object
    data = CONLLU(conllu_file)

    assert len(data) == data_quantity
