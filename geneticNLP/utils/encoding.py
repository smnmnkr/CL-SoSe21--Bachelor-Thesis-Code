from typing import Iterable, Any, Dict


class Encoding:

    """A class which represents a mapping between (hashable) objects
    and unique atoms (represented as ints).
    >>> objects = ["English", "German", "French"]
    >>> enc = Encoding(objects)
    >>> assert "English" == enc.decode(enc.encode("English"))
    >>> assert "German" == enc.decode(enc.encode("German"))
    >>> assert "French" == enc.decode(enc.encode("French"))
    >>> set(range(3)) == set(enc.encode(ob) for ob in objects)
    True
    >>> for ob in objects:
    ...     ix = enc.encode(ob)
    ...     assert 0 <= ix <= enc.obj_num
    ...     assert ob == enc.decode(ix)
    """

    def __init__(self, objects: Iterable[Any]):

        obj_set = {ob for ob in objects}

        self.obj_to_ix: Dict[Any, int] = {}
        self.ix_to_obj: Dict[int, Any] = {}

        for (ix, ob) in enumerate(sorted(obj_set)):
            self.obj_to_ix[ob] = ix
            self.ix_to_obj[ix] = ob

    def encode(self, ob: str) -> int:
        return self.obj_to_ix[ob]

    def decode(self, ix: int) -> str:
        return self.ix_to_obj[ix]

    def __len__(self) -> int:
        return len(self.obj_set)
