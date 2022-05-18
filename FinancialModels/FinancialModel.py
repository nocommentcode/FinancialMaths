from matplotlib import pyplot as plt


class FinancialModel:
    """
    A Financial Model, Can be plotted and simulate paths randomly
    """

    def __init__(self, y_name="", x_name="time"):
        """
        Initialises the model
        :param y_name: name of the y-axis when plotting
        :param x_name: name of the x-axis when plotting
        """
        self.path = None
        self.T = None

        self.x_name = x_name
        self.y_name = y_name

    def plot(self):
        """
        Plots the generated paths, must have called generate_paths() first
        :return: None
        """
        if self.path is None or self.T is None:
            print("Error - tried to plot path but path is None")
            return

        plt.plot(self.T, self.path)
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.show()

    def generate_paths(self, n, t, dt):
        """
        Generates random paths from the model
        :param n: number of paths to generate
        :param t: time to generate samples for
        :param dt: delta in time
        :return: time intervals, list of paths
        """
        pass
