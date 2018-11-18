from code.OA import train
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
    (w0,w1,w2,w3) = param
    m = np.identity(4)
    for i in range(len(y_dev)):
        x = x_dev[i]
        y = y_dev[i]
        fx0 = np.dot(w0, x)
        fx1 = np.dot(w1, x)
        fx2 = np.dot(w2, x)
        fx3 = np.dot(w3, x)
        fx_arr = [fx0,fx1,fx2,fx3]
        d0 = calculate_dist(m[0], fx_arr)
        d1 = calculate_dist(m[1], fx_arr)
        d2 = calculate_dist(m[2], fx_arr)
        d3 = calculate_dist(m[3], fx_arr)
        y_hat = np.argmin([d0, d1, d2, d3])
        if y_hat == y:
            good += 1
        else:
            bad += 1

    print ("precentage is: %f" % (good/(good+bad)))


if __name__ == '__main__':
    x_train, y_train, x_dev, y_dev = load_data()
    (w0,w1,w2,w3) = train(x_train, y_train)
    validate((w0,w1,w2,w3), x_dev, y_dev)
