import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles

def make_spirals(n_points=1000, noise=0.15, rng=None):
    """Gera um dataset de duas espirais entrelaçadas com ruído controlado."""
    rng = np.random.default_rng() if rng is None else rng
    n = n_points // 2
    theta0 = np.sqrt(rng.random(n)) * 2.5 * np.pi  # Ângulo
    r0 = theta0 + 1.5  # Raio
    x0 = (r0 + rng.normal(0.0, noise, n)) * np.cos(theta0)
    y0 = (r0 + rng.normal(0.0, noise, n)) * np.sin(theta0)
    theta1 = np.sqrt(rng.random(n)) * 2.5 * np.pi
    r1 = theta1 + 1.5
    x1 = (r1 + rng.normal(0.0, noise, n)) * np.cos(theta1 + np.pi)
    y1 = (r1 + rng.normal(0.0, noise, n)) * np.sin(theta1 + np.pi)
    X = np.vstack((np.column_stack((x0, y0)), np.column_stack((x1, y1))))
    y = np.hstack((-1 * np.ones(n), np.ones(n)))
    return X, y

make_moons = make_moons
make_circles = make_circles

def plot_data(X, y):
    

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#E0E0E0',"#666464"]),vmin=0.2,vmax=0.8, edgecolors='k')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Data 2D")
    plt.show()