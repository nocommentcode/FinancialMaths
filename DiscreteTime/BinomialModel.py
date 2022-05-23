import numpy as np

from DiscreteTime.ReplicatingPortfolio import ReplicatingPortfolio
from Options.FinancialOption import CallOption


class BinomialNode:
    """
    Represents a Single Step Binomial model
    """

    def __init__(self, S, u, d, b):
        """
        Initialise the node
        :param S: Stock price at node
        :param u: Upper node or None
        :param d: Down node or None
        :param b: Beta value for this node
        """
        self.S = S
        self.u = u
        self.d = d
        self.b = b

    def get_u_factor(self):
        """
        :return: The upper factor of this node
        """
        return self.u.S / self.S

    def get_d_factor(self):
        """
        :return: The down factor of this node
        """
        return self.d.S / self.S

    def replicate_portfolio(self, Cu, Cd):
        """
        Replicates the portfolio
        :param Cu: Option value if stock goes up
        :param Cd: Option value if stock goes down
        :return: (amount of stock, amount of savings)
        """
        u = self.get_u_factor()
        d = self.get_d_factor()

        a = (Cu - Cd) / ((u - d) * self.S)
        b = ((u * Cd) - (d * Cu)) / ((u - d) * self.b)

        return a, b

    def price_by_replication(self, Cu, Cd, return_portfolio=False):
        """
        Find the option price before this node using replication
        :param Cu: Option value if stock goes up
        :param Cd: Option value if stock goes down
        :param return_portfolio: return the (amount of stock, amount of savings) as well as the C0 price
        :return: C0 [, (a,b)]
        """
        a, b = self.replicate_portfolio(Cu, Cd)
        C0 = a * self.S + b
        if return_portfolio:
            return C0, (a, b)
        else:
            return C0

    def get_Cu(self, option):
        """
        Returns the Option value if stock goes up
        :param option: option
        :return: option value
        """
        return option.get_option_payoff(self.u.S)

    def get_Cd(self, option):
        """
        Returns the Option value if stock goes down
        :param option: option
        :return: option value
        """
        return option.get_option_payoff(self.d.S)

    def price_option_by_emm(self, Cu, Cd):
        """
        Find the option price of this node using the EMM
        :param Cu: Option value if stock goes up
        :param Cd: Option value if stock goes down
        :return: C0
        """
        u = self.get_u_factor()
        d = self.get_d_factor()

        p = (self.b - d) / (u - d)

        C0 = (1 / self.b) * (Cu * p + Cd * (1 - p))

        return C0


class MultiPeriodBinomialModel:
    """
    Multi Period Binomial model implementation
    """

    def __init__(self, U, D, S0, Beta):
        """
        Initialises the model
        :param U: scaling factors when price goes up (array, 1 for each timestep)
        :param D: scaling factors when price goes down (array, 1 for each timestep)
        :param S0: Initial stock price
        :param Beta: values of beta (array, 1 for each timestep)
        """
        assert len(U) == len(D) == len(Beta)
        self.T = len(U)
        self.U = U
        self.D = D
        self.S0 = S0
        self.Beta = Beta
        self.root = self.build_node(S0, 0)

    def build_node(self, S, layer):
        """
        Recursive function to build nodes
        :param S: Stock price at this node
        :param layer: the index of the timestep
        :return: the root node
        """
        # stop condition -> end of timesteps
        if layer == self.T:
            return BinomialNode(S, None, None, None)

        # build node
        up_node = self.build_node(S * self.U[layer], layer + 1)
        down_node = self.build_node(S * self.D[layer], layer + 1)
        node = BinomialNode(S, up_node, down_node, self.Beta[layer])

        return node

    def price_option_by_emm(self, option):
        """
        Prices an option recursively using the EMM
        :param option: option to price
        :return: C0
        """
        return self.price_node_by_emm(option, self.root)

    def price_node_by_emm(self, option, node):
        """
        Recursive function to price option by emm
        :param option: option to price
        :param node: current node
        :return: C0 of this node
        """
        # leaf node
        if node.u is not None and node.d is not None:
            if node.u.u is None and node.u.d is None and node.d.u is None and node.d.d is None:
                Cu = option.get_option_payoff(node.u.S)
                Cd = option.get_option_payoff(node.d.S)
                return node.price_option_by_emm(Cu, Cd)

        Cu = self.price_node_by_emm(option, node.u)
        Cd = self.price_node_by_emm(option, node.d)
        return node.price_option_by_emm(Cu, Cd)

    def price_option_by_replication(self, option):
        """
        Prices an option recursively using replication
        :param option: option to price
        :return: C0
        """
        return self.price_node_by_replication(option, self.root)

    def price_node_by_replication(self, option, node):
        """
        Recursive function to price node by replication
        :param option: the option to price
        :param node: the current node
        :return: the C0 price of the node
        """
        # leaf node
        if node.u is not None and node.d is not None:
            if node.u.u is None and node.u.d is None and node.d.u is None and node.d.d is None:
                Cu = option.get_option_payoff(node.u.S)
                Cd = option.get_option_payoff(node.d.S)
                return node.price_by_replication(Cu, Cd)

        Cu = self.price_node_by_replication(option, node.u)
        Cd = self.price_node_by_replication(option, node.d)
        return node.price_by_replication(Cu, Cd)

    def get_replicating_portfolio(self, option):
        """
        Get the replicating portfolio of the option
        :param option: the option to calculate the portfolio for
        :return: the portfolio
        """
        portfolio = ReplicatingPortfolio(self.T)
        _ = self.get_node_replicating_portfolio(option, self.root, portfolio, [])
        return portfolio

    def get_node_replicating_portfolio(self, option, node, portfolio, curr_path):
        """
        Calculates the replicating portfolio recursively
        :param option: option to calculate on
        :param node: the current node
        :param portfolio: the portfolio
        :param curr_path: the current 0s (down) and 1s (up) array representing the current path
        :return: the replicating portfolio
        """
        # leaf node
        if node.u is not None and node.d is not None:
            if node.u.u is None and node.u.d is None and node.d.u is None and node.d.d is None:
                Cu = option.get_option_payoff(node.u.S)
                Cd = option.get_option_payoff(node.d.S)

                C0, (a, b) = node.price_by_replication(Cu, Cd, return_portfolio=True)
                portfolio.set(curr_path, a, b)
                return C0

        Cu = self.get_node_replicating_portfolio(option, node.u, portfolio, [*curr_path, 1])
        Cd = self.get_node_replicating_portfolio(option, node.d, portfolio, [*curr_path, 0])

        C0, (a, b) = node.price_by_replication(Cu, Cd, return_portfolio=True)

        portfolio.set(curr_path, a, b)

        return C0


if __name__ == "__main__":
    K = 1.5
    sigma = 0.15
    T = 5
    callOption = CallOption(T, K, sigma)

    u = [1.4, 1.5, 1.9]
    d = [0.8, 0.7, 1.1]
    beta = [1.1, 1.1 ** 2, 1.1 ** 3]
    S = 1

    model = MultiPeriodBinomialModel(u, d, S, beta)

    emm_price = model.price_option_by_emm(callOption)
    rep_price = model.price_option_by_replication(callOption)
    print(f"Price using emm: ${round(emm_price, 2)}")
    print(f"Price using replicating portfolio: ${round(rep_price, 2)}")

    portfolio = model.get_replicating_portfolio(callOption)
    print("Replicating Portfolio:")
    a, b = portfolio.get([])
    print(f"(a0, b0) : ({a.round(2)}, {b.round(2)})")

    a1d, b1d = portfolio.get([0])
    a1u, b1u = portfolio.get([1])
    print(f"(a1u, b1u) : ({(a1u.round(2)[0])}, {(b1u.round(2)[0])})")
    print(f"(a1d, b1d) : ({(a1d.round(2)[0])}, {(b1d.round(2)[0])})")

    a2dd, b2dd = portfolio.get([0, 0])
    a2du, b2du = portfolio.get([0, 1])
    a2ud, b2ud = portfolio.get([1,  0])
    a2uu, b2uu = portfolio.get([1, 1])
    print(f"(a2uu, b2uu) : ({(a2uu.round(2)[0])}, {(b2uu.round(2)[0])})")
    print(f"(a2ud, b2ud) : ({(a2ud.round(2)[0])}, {(b2ud.round(2)[0])})")
    print(f"(a2du, b2du) : ({(a2du.round(2)[0])}, {(b2du.round(2)[0])})")
    print(f"(a2dd, b2dd) : ({(a2dd.round(2)[0])}, {(b2dd.round(2)[0])})")
