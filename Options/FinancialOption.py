import numpy as np
from matplotlib import pyplot as plt


class FinancialOption:
    """
    Class for a financial option
    """

    def __init__(self, T, K, sigma):
        """
        Initialises the option
        :param T: Time to maturity
        :param K: Strike price
        :param sigma: volatility
        """
        self.T = T
        self.K = K
        self.sigma = sigma
        self.price = None

    def get_params(self):
        """
        :return: Time to maturity, strike price, volatility
        """
        return self.T, self.K, self.sigma


    def plot_price(self, x, prices,names, x_label = "", y_label="price ($)", title=""):
        """
        Plots the option price
        :param x: an array of x axis
        :param prices: an array of prices
        :param names: an array of names of x axis
        :return: None
        """
        for (x, price, name) in zip(x, prices, names):
            plt.plot(x, price, label=name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()

        plt.show()


if __name__ == "__main__":
    T = 1
    t = 0
    St = 300
    r = 0.03
    K = np.arange(230, 240)
    sigma = 0.15
    option = FinancialOption(T, K, sigma)
