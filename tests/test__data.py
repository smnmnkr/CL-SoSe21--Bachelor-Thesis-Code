from beyondGD.data import CONLLU

# NOTE: user dependent file
conllu_file: str = "./data/en_partut-ud-test.conllu"

data_quantity: int = 153


def test_loader():

    # create embedding object
    data = CONLLU(conllu_file)

    assert len(data) == data_quantity
