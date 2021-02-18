import torch

from beyondGD.models.components import MLP, BILSTM

# config
in_size: int = 16
hid_size: int = 8

sent_len: int = 32
batch_len: int = 48

# inputs
x_vec: torch.Tensor = torch.randn(in_size)
x_mat: torch.Tensor = torch.randn(sent_len, in_size)
x_bat: list = [
    torch.randn(sent_len, in_size) for _ in range(batch_len)
]


def test_BILSTM():

    net = BILSTM(
        in_size=in_size,
        hid_size=hid_size,
        depth=1,
        dropout=0.0,
    )

    # forward batch:
    assert len(net.forward(x_bat)) == 2
    assert net.forward(x_bat)[0].size()[0] == batch_len
    assert net.forward(x_bat)[0].size()[1] == sent_len
    assert net.forward(x_bat)[0].size()[2] == hid_size * 2


def test_mlp():

    net = MLP(
        in_size=in_size,
        hid_size=hid_size,
        dropout=0.0,
    )

    # forward vector:
    assert isinstance(net.forward(x_vec), torch.FloatTensor)
    assert net.forward(x_vec).size()[0] == hid_size

    # forward matrix:
    assert isinstance(net.forward(x_mat), torch.FloatTensor)
    assert net.forward(x_mat).size()[0] == sent_len
    assert net.forward(x_mat).size()[1] == hid_size
