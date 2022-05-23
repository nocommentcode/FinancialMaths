import numpy as np
from OptionPricing.BSOptionPricer import BSCallOptionPricer, BSPutOptionPricer
from OptionPricing.MonteCarloOptionPricer import MonteCarloOptionPricer
from Options.FinancialOption import FinancialOption, CallOption, PutOption


def price_option_via_MC(option, t, St, r, mu):
    callOption = CallOption(T, K, sigma)
    putOption = PutOption(T, K, sigma)

    pricer = MonteCarloOptionPricer(mu, 1000, 0.01)
    call_price = pricer.price(t, St, r, callOption)
    put_price = pricer.price(t, St, r, putOption)

    return call_price, put_price


def price_optiion_via_BS(option, t, St, r):
    call_pricer = BSCallOptionPricer()
    call_price = call_pricer.price(t, St, r, option)

    put_pricer = BSPutOptionPricer()
    put_price = put_pricer.price(t, St, r, option)

    return call_price, put_price


def price_option(option, t, St, r, mu):
    put_price, call_price = price_optiion_via_BS(option, t, St, r)
    option.plot_price([K, K], [put_price, call_price], ["Call Option", "Put Option"], x_label="Strike Price ($)",
                      title="Strike price vs option price using BS equation")

    put_price, call_price = price_option_via_MC(option, t, St, r, mu)
    option.plot_price([K, K], [put_price, call_price], ["Call Option", "Put Option"], x_label="Strike Price ($)",
                      title="Strike price vs option price using MC simulation")




if __name__ == "__main__":
    T = 3
    t = 0
    St = 300
    r = 0.03
    mu = 0
    K = np.arange(1, 1000)
    sigma = 0.15
    option = FinancialOption(T, K, sigma)

    price_option(option, t, St, r, mu)
