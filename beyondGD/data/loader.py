from typing import Union

import math

from torch.utils.data import (
    Dataset,
    IterableDataset,
    DataLoader,
)

#
#
#  -------- batch_loader -----------
#
def batch_loader(
    data_set: Union[IterableDataset, Dataset],
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a batch data loader from the given data set.
    """
    return DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        shuffle=shuffle,
        num_workers=num_workers,
    )


#
#
#  -------- adpative_batch_loader -----------
#
def adpative_batch_loader(
    data_set: Union[IterableDataset, Dataset],
    epoch: int,
    batch_size: int = 32,
    batch_double: int = 20,
) -> DataLoader:

    # handle aptative batch size
    batch_size: int = max(
        batch_size, batch_size * int(epoch / batch_double)
    )

    # return batched loader
    return batch_loader(
        data_set,
        batch_size=batch_size,
    )
