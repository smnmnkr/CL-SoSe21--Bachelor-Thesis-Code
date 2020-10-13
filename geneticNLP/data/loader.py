from typing import Union

from torch.utils.data import Dataset, IterableDataset, DataLoader

#
#
#  -------- batch_loader -----------
#
def batch_loader(
    data_set: Union[IterableDataset, Dataset],
    batch_size: bool,
    shuffle=False,
) -> DataLoader:
    return DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        shuffle=shuffle,
    )
