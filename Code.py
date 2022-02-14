# Code for COM3013 Coursework 2
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sympy.combinatorics.graycode import gray_to_bin
from deap import creator, base, tools


def eval_function(x1, x2):
    return np.sin(3.5 * x1 + 1) * np.cos(5.5 * x2)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.hidden2(x))
        x = self.out(x)
        return x

def main():
    # Code for 1.1
    xrange = np.linspace(-1.0, 1.0, 100)
    yrange = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(xrange, yrange)
    Z = eval_function(X, Y)

    fig = plt.figure(figsize=(7, 7))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-1, 1)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    # Code for 1.2
    x1_values = []
    x2_values = []
    y_values = []

    for i in range(1100):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        y = eval_function(x1, x2)
        x1_values.append(x1)
        x2_values.append(x2)
        y_values.append(y)

    x1_train = torch.as_tensor(x1_values[0:1000], dtype=torch.double)
    x2_train = torch.as_tensor(x2_values[0:1000], dtype=torch.double)
    y_train = torch.as_tensor(y_values[0:1000], dtype=torch.double)

    x1_test = torch.as_tensor(x1_values[1000:1100], dtype=torch.double)
    x2_test = torch.as_tensor(x2_values[1000:1100], dtype=torch.double)
    y_test = torch.as_tensor(y_values[1000:1100], dtype=torch.double)

    fig = plt.figure(figsize=(20, 10))

    train = fig.add_subplot(1, 2, 1, projection='3d')
    train.scatter3D(x1_train, x2_train, y_train, color="k")

    test = fig.add_subplot(1, 2, 2, projection='3d')
    test.scatter3D(x1_test, x2_test, y_test, color="k")

    plt.show()




if __name__ == '__main__':
    main()
