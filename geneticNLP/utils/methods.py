from typing import List

import json
import operator
import functools
from functools import wraps

from time import time

import torch
import torch.nn.utils.rnn as rnn

from geneticNLP.utils.types import TT

#
#
# -------- time_track -----------
#
def time_track(func):
    @wraps(func)
    def wrap(*args, **kw):

        t_start = time()
        result = func(*args, **kw)
        t_end = time()

        duration = t_end - t_start

        print(
            f"[--- TIMETRACK || method: {func.__name__} -- time: {duration:2.4f} sec. ---]"
        )

        return result

    return wrap


#
#
#  -------- get_device -----------
#
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


#
#
#  -------- unpack -----------
#
def unpack(pack: rnn.PackedSequence) -> List[TT]:
    """Convert the given packaged sequence into a list of vectors."""
    padded_pack, padded_len = rnn.pad_packed_sequence(
        pack, batch_first=True
    )
    return unpad(padded_pack, padded_len)


#
#
#  -------- unpad -----------
#
def unpad(padded: TT, length: TT) -> List[TT]:
    """Convert the given packaged sequence into a list of vectors."""
    output = []
    for v, n in zip(padded, length):
        output.append(v[:n])
    return output


#
#
#  -------- flatten -----------
#
def flatten(l: list):
    return functools.reduce(operator.iconcat, l, [])


#
#
#  -------- load_json -----------
#
def load_json(path: str) -> dict:
    """Load JSON configuration file."""
    with open(path) as data:
        return json.load(data)
