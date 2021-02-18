import os
import pytest
import torch


from beyondGD.model.postagger import POSstripped

config: dict = {
    "lstm": {
        "in_size": 32,
        "hid_size": 16,
        "depth": 1,
        "dropout": 0.5,
    },
    "score": {"hid_size": 8, "dropout": 0.5},
}

input = torch.rand(4, 32, 32)

save_path: str = "tests/static/stripped_save"


# share fixed POSstripped object across module test
# src: https://docs.pytest.org/en/2.8.7/fixture.html
@pytest.fixture(scope="module")
def POSstripped_fixture():
    return POSstripped(config)


def test_POSstripped_init(POSstripped_fixture):
    assert POSstripped_fixture.config == config


def test_POSstripped_forward(POSstripped_fixture):
    result = POSstripped_fixture.forward(input)

    assert len(result) == 4
    assert result[0].size() == torch.Size([32, 8])


def test_POSstripped_save_load(POSstripped_fixture):

    POSstripped_fixture.save(save_path)

    assert os.path.exists(save_path + ".pickle") == True

    loaded_model = POSstripped.load(save_path)

    assert id(POSstripped_fixture) != id(loaded_model)
    assert POSstripped_fixture.config == loaded_model.config

    for p1, p2 in zip(
        POSstripped_fixture.parameters(),
        loaded_model.parameters(),
    ):
        assert p1.data.ne(p2.data).sum() == 0

    os.remove(save_path + ".pickle")


def test_POSstripped_copy(POSstripped_fixture):

    copy = POSstripped.copy(POSstripped_fixture)

    assert id(POSstripped_fixture) != id(copy)
    assert POSstripped_fixture.config == copy.config

    for p1, p2 in zip(
        POSstripped_fixture.parameters(),
        copy.parameters(),
    ):
        assert p1.data.ne(p2.data).sum() == 0
