
from IO import IO

class TestDomain:
    def __init__(self):
        pass

    def sample_input(self, n:int):
        # sample a bunch of input sequences
        raise NotImplementedError

    def sample_data(self, n:int):
        # This returns a list of pairs of (input,output)
        for inp in self.sample_input(n):
            yield IO(inp, self.call(inp))

    # calls to the correct output function
    def call(self, *args, **kwargs):
        raise NotImplementedError

    