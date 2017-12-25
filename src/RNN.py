import numpy as np

class RNN:
    def __init__(self):
        # hidden state self.h is initialized with the zero vector
        self.h = np.zeros()


    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))

        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y
