import numpy as np

from FinancialModels.FinancialModel import FinancialModel


class BM(FinancialModel):
    """
    A Brownian Motion model
    """

    def __init__(self, b0=0):
        super().__init__(y_name="Bt")
        self.b0 = b0

    def generate_paths(self, n, time, dt):
        # time intervals
        T = np.arange(0, time, step=dt)

        # generate random samples from a normal distribution
        # N = np.random.normal(loc=0, scale=time * np.sqrt(dt), size=(T.size, n))
        N = np.random.normal(loc=0, scale=np.sqrt(dt), size=(T.size, n))

        # calculate brownian increments and build path iteratively
        B = np.zeros((T.size, n))
        for i, (Ti, Ni) in enumerate(zip(T, N)):
            if i == 0:
                B[i, :] = np.array([self.b0 for _ in range(n)])
            else:
                B[i, :] = B[i - 1, :] + Ni

        self.path, self.T = B, T

        return T, self.path


if __name__ == "__main__":
    bm = BM()
    bm.generate_paths(100, 2, 1 / 365)
    bm.plot()
