import pytest
import torch

from geneticNLP.embeddings import FastText, Untrained

# NOTE: user dependent file
fasttext_file: str = "./data/cc.en.32.bin"

# config
dimension: int = 32

word: str = "connection"
sent: list = [
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
    return FastText(fasttext_file, dimension=dimension)


# share fixed Untrained object across module test
@pytest.fixture(scope="module")
def ut_fixture():
    return Untrained(sent, dimension)


def test_FastText_init(fs_fixture):
    # check dimensionality of the embeddings
    assert fs_fixture.dimension == dimension


def test_FastText_forward_word(fs_fixture):
    # test: forward single word
    forward_word = fs_fixture.forward_tok(word)

    assert isinstance(forward_word, torch.FloatTensor)
    assert forward_word.size()[0] == dimension


def test_FastText_fordward_sentence(fs_fixture):
    # test: forward multiply words/sentence
    forward_sent = fs_fixture.forward_sent(sent)

    assert isinstance(forward_sent, torch.FloatTensor)
    assert forward_sent.size()[0] == len(sent)
    assert forward_sent.size()[1] == dimension


def test_Untrained_init(ut_fixture):
    # check dimensionality of the embeddings
    assert ut_fixture.dimension == dimension


def test_Untrained_forward_word(ut_fixture):
    # test: forward single word
    forward_word = ut_fixture.forward_tok(word)

    assert isinstance(forward_word, torch.FloatTensor)
    assert forward_word.size()[0] == dimension


def test_Untrained_fordward_sentence(ut_fixture):
    # test: forward multiply words/sentence
    forward_sent = ut_fixture.forward_sent(sent)

    assert isinstance(forward_sent, torch.FloatTensor)
    assert forward_sent.size()[0] == len(sent)
    assert forward_sent.size()[1] == dimension
