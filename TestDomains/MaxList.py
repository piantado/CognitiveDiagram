import random
import numpy
from TestDomain import TestDomain

class MaxList(TestDomain):
    alphabet=[0,1,2,3,4,5,6,7,8,9]

    def __init__(self):
        pass

    def sample_input(self, n:int):
        for i in range(n):
            L = random.randint(1,10) # a length of a list
            yield list(numpy.random.choice(self.alphabet, size=L))

    def call(self, lst:list):
        # NOTE that call should return a sequence of outputs that has the same length as list

        return (["."] * (len(lst)-1)) + [max(lst)]

if __name__ == "__main__":

    D = MaxList()

    for k in D.sample_data(10):
        print(k)
