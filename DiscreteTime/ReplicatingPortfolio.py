import numpy as np


class ReplicatingPortfolio:
    """
    A store of (a, b) for each step in a binomial model
    """
    def __init__(self, T):
        self.T = T
        self.As = [np.zeros(1)] + [np.zeros([2 for _ in range(path_length + 1)]) for path_length in range(T -1)]
        self.Bs = [np.zeros(1)] + [np.zeros([2 for _ in range(path_length + 1)]) for path_length in range(T-1)]

    def set(self, path, a, b):
        """
        Sets the value of a and b in the portfolio for a given path
        :param path: array of 0s and 1s corresponding to up/down journey through binomial model
        :param a: amount of stock
        :param b: amount of savings account
        :return:
        """
        if path == []:
            self.As[0][0] = a
            self.Bs[0][0] = b
        self.As[len(path)][path] = a
        self.Bs[len(path)][path] = b

    def get(self, path):
        """
        gets the value of a and b in the portfolio for a given path
        :param path: array of 0s and 1s corresponding to up/down journey through binomial model
        :return: (amount of stock, amount of savings)
        """
        if path == []:
            return self.As[0][0], self.Bs[0][0]

        return self.As[len(path)][path], self.Bs[len(path)][path]

