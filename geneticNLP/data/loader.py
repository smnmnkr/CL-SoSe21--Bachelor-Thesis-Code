from conllu import parse_incr


class Loader:
    """Module for CONLLU Data Loading."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, data_path):

        # save data
        self.data = list(self.load_data(data_path))

    #
    #
    #  -------- load_data -----------
    #
    def load_data(self, data_path):
        data_file = open(data_path, encoding="utf-8")
        yield from parse_incr(data_file)

    #
    #
    #  -------- data_quantity -----------
    #
    def data_quantity(self) -> int:
        return len(self.data)
