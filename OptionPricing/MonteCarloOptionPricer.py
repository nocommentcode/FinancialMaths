import numpy as np

from FinancialModels.GeometricBrownianMotion import GBM
from OptionPricing.OptionPricer import OptionPricer
from Options.FinancialOption import FinancialOption


class MonteCarloOptionPricer(OptionPricer):
    """
    Prices options using
    """
    def __init__(self, mu, n, dt):
        """
        Initialises a MC option pricer
        :param mu: the drift term
        :param n: number of samples
        :param dt: time delta
        """
        self.mu = mu
        self.n = n
        self.dt = dt

    def get_option_payoff(self, option, stock_price):
        """
        Returns the option payoff
        :param option: the option
        :param stock_price: the final stock price
        :return: payoff
        """
        pass

    def price(self, t, St, r, option):
        T, K, sigma = option.get_params()
        gbm = GBM(self.mu, sigma, St)
        stock_price = gbm.generate_paths(self.n, T-t, self.dt)
        final_stock_price = stock_price[-1, :].mean()
        print(final_stock_price)
        option_price = self.get_option_payoff(option, final_stock_price)
        return option_price


class MCCallOptionPricer(MonteCarloOptionPricer):
    def get_option_payoff(self, option, stock_price):
        return np.maximum(stock_price - option.K, 0)


class MCPutOptionPricer(MonteCarloOptionPricer):
    def get_option_payoff(self, option, stock_price):
        return np.maximum(option.K - stock_price, 0)


if __name__ == "__main__":
    T = 3
    t = 0
    St = 300
    r = 0.03
    # K = 250
    mu = 0
    K = np.arange(1, 1000)
    sigma = 0.15
    option = FinancialOption(T, K, sigma)

    Cpricer = MCCallOptionPricer(mu, 1000, 0.01)
    Cprice = Cpricer.price(t, St, r, option)

    Ppricer = MCPutOptionPricer(mu, 1000, 0.01)
    Pprice = Ppricer.price(t, St, r, option)

    option.plot_price([K, K], [Cprice, Pprice], ["Call Option", "Put Option"], x_label="Strike Price ($)", title="Strike price vs option price")
