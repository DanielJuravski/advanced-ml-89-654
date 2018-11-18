from code.AP import train, set_learning_params
from code.utils import load_data
import numpy as np


def calculate_dist(m_row, fx_arr):
    val = 0
    for i in range(len(fx_arr)):
        val+= (1 - np.sign(m_row[i]*fx_arr[i])) / 2

    return val


def validate(param, x_dev, y_dev):
    good = 0
    bad = 0
    (w01, w02, w03, w12, w13, w23) = param
    m = np.matrix([[1,1,1,0,0,0],
                   [-1,0,0,1,1,0],
                   [0,-1,0,-1,0,1],
                   [0,0,-1,0,-1,-1]])
    for i in range(len(y_dev)):
        x = x_dev[i]
        y = y_dev[i]
        y_hat = predict(m, w01, w02, w03, w12, w13, w23, x)
        if y_hat == y:
            good += 1
        else:
            bad += 1

    print ("precentage is: %f" % (good/(good+bad)))


def predict(m, w01, w02, w03, w12, w13, w23, x):
    fx01 = np.dot(w01, x)
    fx02 = np.dot(w02, x)
    fx03 = np.dot(w03, x)
    fx12 = np.dot(w12, x)
    fx13 = np.dot(w13, x)
    fx23 = np.dot(w23, x)
    fx_arr = [fx01, fx02, fx03, fx12, fx13, fx23]

    d0 = calculate_dist((m[0]).tolist()[0], fx_arr)
    d1 = calculate_dist((m[1]).tolist()[0], fx_arr)
    d2 = calculate_dist((m[2]).tolist()[0], fx_arr)
    d3 = calculate_dist((m[3]).tolist()[0], fx_arr)

    y_hat = np.argmin([d0, d1, d2, d3])
    return y_hat


if __name__ == '__main__':
    set_learning_params(0.5, 0.05)
    x_train, y_train, x_dev, y_dev = load_data()
    (w01, w02, w03, w12, w13, w23) = train(x_train, y_train)
    validate((w01, w02, w03, w12, w13, w23), x_dev, y_dev)

