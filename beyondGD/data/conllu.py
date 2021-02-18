from conllu import parse_incr

import torch.utils.data as data

from beyondGD.utils.types import Token


class CONLLU(data.IterableDataset):
    """Module for CONLLU Data Loading."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, data_path):

        # save data, taglist
        self.data = list(self.load_data(data_path))
        self.taglist: set = {
            tok.pos for sent in self.data for tok in sent
        }

    #
    #
    #  -------- load_data -----------
    #
    def load_data(self, data_path):

        with open(data_path, encoding="utf-8") as data_file:
            for tok_list in parse_incr(data_file):

                sent: list = []

                for tok in tok_list:
                    form = tok["form"]
                    upos = tok["upostag"]
                    head = tok["head"]

                    sent.append(Token(form, upos, head))

                yield sent

    #
    #
    #  -------- __iter__ -----------
    #
    def __iter__(self):
        yield from self.data

    #
    #
    #  -------- __len__ -----------
    #
    def __len__(self) -> int:
        return len(self.data)
