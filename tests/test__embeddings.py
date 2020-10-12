import pytest
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


# share fixed FastText object across module test
# src: https://docs.pytest.org/en/2.8.7/fixture.html
@pytest.fixture(scope="module")
def fs_fixture():
    return FastText(fasttext_file)


def test_FastText_init(fs_fixture):

    # check dimensionality of the embeddings
    assert fs_fixture.embedding_dim() == fasttext_dim

    # check dimensionality of the embeddings
    assert fs_fixture.embedding_dim() == fasttext_dim


def test_FastText_forward_word(fs_fixture):

    # test: forward single word
    forward_word = fs_fixture.forward(test_word)

    assert isinstance(forward_word, torch.FloatTensor)
    assert forward_word.size()[0] == fasttext_dim


def test_FastText_fordward_sentence(fs_fixture):

    # test: forward multiply words/sentence
    forward_sent = fs_fixture.forwards(test_sent)

    assert isinstance(forward_sent, torch.FloatTensor)
    assert forward_sent.size()[0] == len(test_sent)
    assert forward_sent.size()[1] == fasttext_dim
