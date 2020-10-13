import random
import copy

#
#
#  -------- mutate -----------
#
def mutate(
    parent_network,
    mutation_rate: float = 0.02,
):

    #  -------- _mutate -----------
    #
    def _mutate(param):
        return param + (mutation_rate * random.randint(-1, 1))

    child_network = copy.deepcopy(parent_network)

    for layer in child_network.parameters():

        if len(layer.shape) == 4:
            print("shape not mutated")

        if len(layer.shape) == 3:
            print("shape not mutated")

        # mutate matrices : [1 x n]
        if len(layer.shape) == 2:
            for i0 in range(layer.shape[0]):
                for i1 in range(layer.shape[1]):
                    layer[i0][i1] = _mutate(layer[i0][i1])

        # mutate vectors : [1]
        if len(layer.shape) == 1:
            for i0 in range(layer.shape[0]):
                layer[i0] = _mutate(layer[i0])

    return child_network
