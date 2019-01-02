import random
import sys
from datetime import datetime
from random import shuffle

import numpy as np

import utils

random.seed(3)
np.random.seed(3)
EPOCHS = 30
LR = 1


def init_perceptron(x_size, y_size):
    eps = 0#np.sqrt(6) / np.sqrt(x_size + y_size)
    w = np.random.uniform(-eps, eps, (y_size, x_size))
    b = np.random.uniform(-eps, eps, y_size)
    global w_accumelator, b_accumelator, t_counter
    w_accumelator = np.zeros((y_size, x_size))
    b_accumelator = np.zeros(y_size)
    t_counter = 0
    return (w, b)


def train_perceptron(perceptron, train_data):
    w,b = perceptron
    for epoch in range(EPOCHS):
        shuffle(train_data)
        good = bad = 0.0
        for e_i, example in enumerate(train_data):
            # print("iter %d" % e_i)
            x = np.array(example[utils.PIXEL_VECTOR])
            y = utils.L2I(example[utils.LETTER])
            mul = np.dot(w, x) + b
            y_hat = np.argmax(mul)
            global w_accumelator, b_accumelator, t_counter
            t_counter += 1
            if y != y_hat:
                bad += 1
                for i, row in enumerate(w):
                    if i == y:
                        w[i] += LR * x
                        b[i] += LR
                    elif i==y_hat:
                        w[i] -= LR * x
                        b[i] -= LR

            else:
                good += 1

            w_accumelator += w
            b_accumelator += b
        perc = (good / (good + bad)) * 100
        print("epoch no: %d acc = %.2f" % (epoch, perc))
    return (w, b)


def apply_average_update(w, w_accumelator, b, b_accumelator, t_counter):
    wdelta = w_accumelator / float(t_counter)
    bdelta = b_accumelator / float(t_counter)
    w += wdelta
    b += bdelta
    return w,b


def test(perceptron, test_data):
    w,b = perceptron
    good = bad = 0.0
    pred_results = []
    for example in test_data:
        x = np.array(example[utils.PIXEL_VECTOR])
        y = utils.L2I(example[utils.LETTER])
        mul = np.dot(w, x) + b
        y_hat = np.argmax(mul)
        pred_results.append(y_hat)
        if y != y_hat:
            bad += 1
        else:
            good += 1

    perc = (good / (good + bad)) * 100
    print("==========Test accuracy = %.2f==========" % perc)
    return pred_results


if __name__ == '__main__':
    train_input_file = "data/letters.train.data"
    test_input_file = "data/letters.test.data"
    if len(sys.argv) == 3:
        train_input_file = sys.argv[1]
        test_input_file = sys.argv[2]

    print("started at %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    x_size = 8*16
    y_size = 26

    w_accumelator = None
    b_accumelator = None
    t_counter = 0

    train_data = utils.load_data(train_input_file)
    perceptron = init_perceptron(x_size, y_size)
    w,b = perceptron
    w0 = np.copy(w)
    b0 = np.copy(b)
    perceptron = train_perceptron(perceptron, train_data)
    wf,bf = apply_average_update( w0, w_accumelator, b0, b_accumelator, t_counter)

    test_data = utils.load_data(test_input_file)
    test_results = test((wf,bf), test_data)
    utils.print_results_by_letter(test_results, "multiclass.pred")
    print("finished at %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
