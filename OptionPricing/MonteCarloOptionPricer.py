import numpy as np

from FinancialModels.GeometricBrownianMotion import GBM
from OptionPricing.OptionPricer import OptionPricer
from Options.FinancialOption import FinancialOption, CallOption, PutOption


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


    def price(self, t, St, r, option):
        T, K, sigma = option.get_params()
        gbm = GBM(self.mu, sigma, St)
        stock_price = gbm.generate_paths(self.n, T-t, self.dt)
        payoffs = option.get_option_payoff(stock_price[-1, :]).mean(axis=1)
        discounted = np.exp(-r * (T - t)) * payoffs
        return discounted




if __name__ == "__main__":
    T = 15
    t = 0
    St = 300
    r = 0.03
    # K = 250
    mu = 0
    K = np.arange(1, 1000)
    sigma = 0.15
    callOption = CallOption(T, K, sigma)
    putOption = PutOption(T, K, sigma)

    pricer = MonteCarloOptionPricer(mu, 1000, 0.01)
    Cprice = pricer.price(t, St, r, callOption)
    Pprice = pricer.price(t, St, r, putOption)

    callOption.plot_price([K, K], [Cprice, Pprice], ["Call Option", "Put Option"], x_label="Strike Price ($)", title="Strike price vs option price")

