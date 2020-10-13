from typing import NamedTuple

import torch


#  -------- Tensor type -----------
#
TT = torch.TensorType


#  -------- Token -----------
#
class Token(NamedTuple):
    word: str
    pos: str
    head: int
