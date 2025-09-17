


class TestDomain:
    def __init__(self):
        pass

    def sample_input(self, n:int):
        raise NotImplementedError

    def sample_data(self, n:int):
        # a list of pairs (input,output)
        raise NotImplementedError

    # calls to the correct output function
    def call(self, *args, **kwargs):
        raise NotImplementedError

    