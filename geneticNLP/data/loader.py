from conllu import parse


class Loader:
    """Module for CONLLU Data Loading."""

    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self, data_path):

        # save data
        self.data = self.load_data(data_path)

    #
    #
    #  -------- load_model -----------
    #
    def load_data(self, data_path):
        return parse(data_path)

    #
    #
    #  -------- embedding_dim -----------
    #
    def data_quantity(self) -> int:
        return self.model.get_dimension()
