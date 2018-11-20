from tournament import train, set_learning_params
from utils import load_data
import numpy as np


def calculate_dist(m_row, fx_arr):
    def loss(a,b):
        return np.max([0, (1-(a * b))])
    points_loss = [loss(x[0], x[1]) for x in list(zip(m_row, fx_arr))]
    return np.sum(points_loss)


def validate(param, x_dev, y_dev):
    good = 0
    bad = 0
    (w01v23, w0v1, w2v3) = param
    m = np.matrix([[1,1,0],
                   [1,-1,0],
                   [-1,0,1],
                   [-1,0,-1]])
    for i in range(len(y_dev)):
        x = x_dev[i]
        y = y_dev[i]
        y_hat = predict(m,(w01v23, w0v1, w2v3), x)
        if y_hat == y:
            good += 1
        else:
            bad += 1

    print ("precentage is: %f" % (good/(good+bad)))


def predict(m, weights, x):
    (w01v23, w0v1, w2v3) = weights
    fx01v23 = np.dot(w01v23, x)
    fx0v1 = np.dot(w0v1, x)
    fx2v3 = np.dot(w2v3, x)
    fx_arr = [fx01v23, fx0v1, fx2v3]

    d0 = calculate_dist((m[0]).tolist()[0], fx_arr)
    d1 = calculate_dist((m[1]).tolist()[0], fx_arr)
    d2 = calculate_dist((m[2]).tolist()[0], fx_arr)
    d3 = calculate_dist((m[3]).tolist()[0], fx_arr)

    y_hat = np.argmin([d0, d1, d2, d3])
    return y_hat


if __name__ == '__main__':
    set_learning_params(0.5, 0.05)
    x_train, y_train, x_dev, y_dev = load_data()
    (w01v23, w0v1, w2v3) = train(x_train, y_train)
    validate((w01v23, w0v1, w2v3), x_dev, y_dev)
