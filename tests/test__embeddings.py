import torch

from geneticNLP.embeddings import FastText

# NOTE: user dependent file
fasttext_file: str = "./data/fasttext--cc.en.300.bin"

# config
fasttext_dim: int = 300

test_word: str = "connection"
test_sent: list = [
    "verifies",
    "that",
    "the",
    "account",
    "name",
    "and",
    "password",
]


def test_FastText():

    # create embedding object
    embedder = FastText(fasttext_file)

    # check dimensionality of the embeddings
    assert embedder.embedding_dim() == fasttext_dim

    # check dimensionality of the embeddings
    assert embedder.embedding_dim() == fasttext_dim

    # test: forward single word
    forward_word = embedder.forward(test_word)

    assert isinstance(forward_word, torch.FloatTensor)
    assert forward_word.size()[0] == fasttext_dim

    # test: forward multiply words/sentence
    forward_sent = embedder.forwards(test_sent)

    assert isinstance(forward_sent, torch.FloatTensor)
    assert forward_sent.size()[0] == len(test_sent)
    assert forward_sent.size()[1] == fasttext_dim
