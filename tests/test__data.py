from geneticNLP.data import Loader

# NOTE: user dependent file
conllu_file: str = "./data/universal-dependencies--en-dev.conllu"

data_quantity: int = 2002


def test_loader():

    # create embedding object
    loader = Loader(conllu_file)

    assert loader.data_quantity() == data_quantity

    assert loader.get1() == ""
