import numpy as np

import matplotlib.pyplot as plt


def linear(x) -> float:
    return 1 - x


def sqrt(x) -> float:
    return 1 - np.sqrt(x)


def inverse_logistic(x, grow_rate: int = 3) -> float:
    return 1 / (1 + (x / (1 - x)) ** grow_rate)


def inverse_sigmoid(x) -> float:
    return 1 - (0.5 * (1 + np.sin((x * np.pi) - (np.pi / 2))))


fig = plt.figure()
ax = plt.axes()


x = np.linspace(0, 1, 1000)

for func in [linear, sqrt, inverse_sigmoid, inverse_logistic]:
    ax.plot(x, func(x), label=func.__name__)

ax.legend()
plt.savefig("mutation_rate_scale_functions")
