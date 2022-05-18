import numpy as np
from scipy.stats import norm

from OptionPricing.OptionPricer import OptionPricer
from Options.FinancialOption import FinancialOption


class BSOptionPricer(OptionPricer):
    """
    A Black Scholes Option Pricer
    Prices the Option via the BS formulat
    """

    def h_t(self, St, t, r, T, K, sigma):
        """
        Calculates h(t) = (ln(St / K) + (r + 0.5 sigma ^2)(T-t) ) / sigma sqrt(T - t)
        :param St: Stock price at time t
        :param t: time t
        :return: h(t)
        """

        numerator = np.log(St / K) + (r + 0.5 * sigma ** 2) * (T - t)
        denominator = sigma * np.sqrt(T - t)
        return numerator / denominator



class BSCallOptionPricer(BSOptionPricer):
    """
    BS model option pricer
    Ct = St phi(ht) - exp(r(T-t)) K phi(ht - sigma sqrt(T - t))
    """

    def price(self, t, St, r, option):
        T, K, sigma = option.get_params()

        ht = self.h_t(St, t, r, T, K, sigma)

        Ct = St * norm.cdf(ht) - np.exp(-r * (T - t)) * K * norm.cdf(ht - sigma * np.sqrt(T - t))
        return Ct


class BSPutOptionPricer(BSOptionPricer):
    """
    BS model option pricer
    Ct =  exp(r(T-t)) K phi(sigma sqrt(T - t) - ht ) - St phi(-ht)
    """


    def price(self, t, St, r, option):
        T, K, sigma = option.get_params()

        ht = self.h_t(St, t, r, T, K, sigma)

        Ct = np.exp(-r * (T - t)) * K * norm.cdf(sigma * np.sqrt(T - t) - ht) - St * norm.cdf(-ht)
        return Ct


if __name__ == "__main__":
    T = 3
    t = 0
    St = 300
    r = 0.03
    # K = 250
    K = np.arange(1, 1000)
    sigma = 0.15
    option = FinancialOption(T, K, sigma)

    Cpricer = BSCallOptionPricer()
    Ppricer = BSPutOptionPricer()
    Cprice = Cpricer.price(t, St, r, option)
    Pprice = Ppricer.price(t, St, r, option)

    option.plot_price([K, K], [Cprice, Pprice], ["Call Option", "Put Option"], x_label="Strike Price ($)", title="Strike price vs option price")
