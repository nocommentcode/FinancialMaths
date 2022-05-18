import numpy as np

from FinancialModels.BrownianMotion import BM
from FinancialModels.FinancialModel import FinancialModel


class GBM(FinancialModel):
    """
    A Geometric Brownian Motion model
    dSt = aSt dt + b St dBt
    St = s0 exp((a - 0.5 b^2)t + b Bt)
    """

    def __init__(self, mu, sigma, s0=1):
        """
        Initialises the GBM
        :param mu: the drift coefficient
        :param sigma: the volatility coefficient
        :param s0: the initial stock price
        """
        super().__init__(y_name="St")
        self.mu = mu
        self.s0 = s0
        self.sigma = sigma

    def generate_paths(self, n, time, dt):
        # generate time intervals and brownian motion
        T, B = BM(0).generate_paths(n, time, dt)

        # initial stock price is the same for all samples
        S0 = np.array([self.s0 for _ in range(n)])

        # drift = (a - 0.5 b^2) x T
        drift = (self.mu - (self.sigma ** 2 / 2)) * T
        # n_drift is drift replicated n times (in order to perform matrix addition
        n_drift = np.repeat(drift[..., np.newaxis], n, axis=1)

        # sigma = b x Bt
        sigma = self.sigma * B

        # St = s0 exp((a - 0.5 b^2)t + b Bt)
        self.path = S0 * np.exp(n_drift + sigma)
        self.T = T

        return self.path


if __name__ == "__main__":
    bm = GBM(0.3, 0.4, 40)
    bm.generate_paths(100, 3, 1/25)
    bm.plot()
