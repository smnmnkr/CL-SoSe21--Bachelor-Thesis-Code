import torch

from geneticNLP.neural.nn import MLP, BILSTM

# config
in_size: int = 16
out_size: int = 8

sent_len: int = 32
batch_len: int = 48

# inputs
x_vec: torch.Tensor = torch.randn(in_size)
x_mat: torch.Tensor = torch.randn(sent_len, in_size)
x_bat: list = [torch.randn(sent_len, in_size) for i in range(batch_len)]
x_pack = None


def test_BILSTM():
    global x_pack

    net = BILSTM(
        in_size=in_size,
        out_size=out_size,
        depth=2,
        dropout=0.0,
    )

    # TODO: forward matrix:

    # forward batch:
    assert net.forward(x_bat).data.size()[0] == sent_len * batch_len
    assert net.forward(x_bat).data.size()[1] == out_size * 2

    # retrieve pack for MLP, BIAFFINE tests
    x_pack = net.forward(x_bat)


def test_mlp():

    net = MLP(
        in_size=in_size,
        hid_size=int(in_size / 2),
        out_size=out_size,
        dropout=0.0,
    )

    # forward vector:
    assert isinstance(net.forward(x_vec), torch.FloatTensor)
    assert net.forward(x_vec).size()[0] == out_size

    # forward matrix:
    assert isinstance(net.forward(x_mat), torch.FloatTensor)
    assert net.forward(x_mat).size()[0] == sent_len
    assert net.forward(x_mat).size()[1] == out_size
