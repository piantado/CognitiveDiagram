

class IO:
    def __init__(self, i:list, o:list):
        assert(len(i) == len(o))
        self.input = i
        self.output = o

    def __str__(self):
        return "IO[%s -> %s]" % (self.input,self.output)