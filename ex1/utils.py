import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


def load_data():
    # read data
    mnist = fetch_mldata("MNIST original", data_home="./data")
    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    # suffle examples
    x, y = shuffle(x, y, random_state=1)

    data_len = len(x)
    train_dev_ratio = round(data_len * 0.8)
    x_train, y_train = x[:train_dev_ratio], y[:train_dev_ratio]
    x_dev, y_dev = x[train_dev_ratio:], y[train_dev_ratio:]

    print("Data was loaded.")

    return x_train, y_train, x_dev, y_dev


def init_param(x_data):
    # create W with dims of feat*labels
    numOfFeat = len(x_data)
    W = np.random.randn(numOfFeat)
    return W

def calculate_dist_loss(m_row, fx_arr):
    def loss(a,b):
        return np.max([0, (1-(a * b))])
    points_loss = [loss(x[0], x[1]) for x in list(zip(m_row, fx_arr))]
    return np.sum(points_loss)


def calculate_dist_hamming(m_row, fx_arr):
    def compare_sign (a,b):
        return (1 - np.sign(a * b)) / 2
    differences = [compare_sign(x[0], x[1]) for x in list(zip(m_row, fx_arr))]
    return np.sum(differences)


def print_results(results, path):
    f = open(path,"w")
    for y in results:
        f.write("%d\n" % y)
    f.close()