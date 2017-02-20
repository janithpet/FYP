import numpy as np

class t1:
    def __init__(self, x, t2):

        self.x = x
        self.reward_function = t2

    def f(self):
        self.x += 1

    def method(self):
        for i in range(2):
            self.f()
            print self.reward_function.x
