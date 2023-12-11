import numpy as np
from matplotlib.colors import LogNorm

from prettyPlot.plotting import plt, pretty_multi_contour


def test_log_cbar():
    N = 100
    X, Y = np.mgrid[-3 : 3 : complex(0, N), -2 : 2 : complex(0, N)]

    Z1 = np.exp(-(X**2) - Y**2)
    Z2 = np.exp(-((X * 10) ** 2) - (Y * 10) ** 2)
    Z = Z1 + 50 * Z2

    pretty_multi_contour(
        [X[:, 0], X[:, 0]], [Z, Z], ybound=[0, 1], log_scale_list=[False, True]
    )


if __name__ == "__main__":
    test_log_cbar()
    plt.show()
