import numpy as np

from prettyPlot.plotsUtil import *


def test_1d(verbose=False):
    x = np.linspace(0, 1, 10)
    y = x**2

    fig = plt.figure()
    plt.plot(x, y, linewidth=3, color="k")
    prettyLabels("x", "y", 14)
    if verbose:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    test_1d(True)
