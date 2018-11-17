from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import numpy as np





def load_data():
    # read data
    mnist = fetch_mldata("MNIST original", data_home="./data")
    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    # suffle examples
    x, y = shuffle(x, y, random_state=1)

    data_len = len(x)
    train_dev_ratio = round(data_len * 0.9)
    x_train, y_train = x[:train_dev_ratio], y[:train_dev_ratio]
    x_dev, y_dev = x[train_dev_ratio:], y[train_dev_ratio:]

    print("Data was loaded.")

    return x_train, y_train, x_dev, y_dev


def init_param(x_data, y_data):
    # create W with dims of feat*labels
    numOfFeat = len(x_data[0])
    numOfLabel = len(y_data[0])
    W = np.random.randn(numOfFeat, numOfLabel)

    return W


def one_VS_all_loss(X, Y, W):

    return W


if __name__ == '__main__':
    X_train, Y_train, X_dev, Y_dev = load_data()
    W_init = init_param(X_train, Y_train)

    W_trained = one_VS_all_loss(X_train, Y_train, W_init)
