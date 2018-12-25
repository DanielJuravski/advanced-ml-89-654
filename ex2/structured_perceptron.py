from datetime import datetime
import random
from random import shuffle

import utils
import numpy as np

random.seed(3)
np.random.seed(3)
EPOCHS = 70
LR = 0.0001


def init_perceptron(x_size, y_size):
    eps = np.sqrt(6) / np.sqrt(x_size + y_size)
    w = np.random.uniform(-eps, eps, y_size *x_size)
    b = np.random.uniform(-eps, eps, y_size)
    return (w,b)


def phi(x, y_candidate):
    x_size = len(x)
    phi_xy = np.zeros(x_size * y_size)
    start = y_candidate*x_size
    phi_xy[start:start+x_size] = x
    return phi_xy


def train_perceptron(perceptron, train_data, y_size):
    w, b = perceptron
    for epoch in range(EPOCHS):
        print("epoch: %d started at %s" %(epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        good = bad = 0.0
        shuffle(train_data)
        for example in train_data:
            x = np.array(example[utils.PIXEL_VECTOR])
            y = utils.L2I(example[utils.LETTER])

            phis = np.empty((y_size, (y_size*x_size)))
            for y_candidate in range(y_size):
                phis[y_candidate] =  phi(example[utils.PIXEL_VECTOR], y_candidate)

            results = np.dot(phis, w) + b
            y_hat = int(np.argmax(results))
            w = w + (LR * (phis[y] - phis[y_hat]))
            b[y] = b[y] + LR
            b[y_hat] = b[y_hat] - LR
            if y != y_hat:
                bad += 1
            else:
                good += 1

        perc = (good / (good + bad)) * 100
        print("epoch no: %d acc = %.2f" % (epoch, perc))
    return (w, b)


def test(perceptron, test_data, y_size):
    w, b = perceptron
    good = bad = 0.0
    for example in test_data:
        x = np.array(example[utils.PIXEL_VECTOR])
        y = utils.L2I(example[utils.LETTER])
        phis = [phi(x, y_cand, y_size) for y_cand in range(y_size)]
        results = [np.dot(w, phi_xy) + b[i] for i, phi_xy in enumerate(phis)]
        y_hat = int(np.argmax(results))
        if y != y_hat:
            bad += 1
        else:
            good += 1

    perc = (good / (good + bad)) * 100
    print("==========Test accuracy = %.2f==========" % perc)
    return perc


if __name__ == '__main__':
    x_size = 8*16
    y_size = 26
    train_data = utils.load_data("data/letters.train.data")
    perceptron = init_perceptron(x_size, y_size)
    perceptron = train_perceptron(perceptron, train_data, y_size)
    test_data = utils.load_data("data/letters.test.data")
    test_acc = test(perceptron, test_data, y_size)