from typing import Tuple, Any, Iterable, Callable

from geneticNLP.data import batch_loader


#
#
#  -------- _accuracy -----------
#
def accuracy(
    predict: Callable,
    data_set: Iterable[Tuple[Any, Any]],
    batch_size: int = 32,
) -> float:
    """Generic accuracy calculation function.

    Arguments:
        * data_set: iterable of input/output pairs
        * predict: function which predicts outputs for the given
            batch of inputs
    """
    k: float = 0.0
    n: float = 0.0

    # We load the dataset in batches to speed up the calculation
    for batch in batch_loader(data_set, batch_size=batch_size):

        # Unzip input and outputs
        inputs, golds = zip(*batch)

        # Tag all the sentences
        predictions = predict(inputs)

        # Process the predictions and compare with the gold labels
        for pred, gold in zip(predictions, golds):
            for (p, g) in zip(pred, gold):
                if p == g:
                    k += 1.0
                n += 1.0

    return k / n
