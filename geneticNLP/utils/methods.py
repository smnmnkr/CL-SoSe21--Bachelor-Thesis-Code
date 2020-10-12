from functools import wraps
from time import time

import torch


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

        return result, duration

    return wrap


#
#
#  -------- get_device -----------
#
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
