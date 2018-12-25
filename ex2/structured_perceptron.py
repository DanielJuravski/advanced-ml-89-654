import random
import sys
from datetime import datetime

import numpy as np

import utils
from utils import get_results

random.seed(3)
np.random.seed(3)
EPOCHS = 70
LR = 0.0001


def init_perceptron():
    eps = np.sqrt(6) / np.sqrt(x_size + y_size)
    w = np.random.uniform(-eps, eps, y_size *x_size)
    b = np.random.uniform(-eps, eps, y_size)
    return w,b


def phi_xi_yi(x, y_candidate):
    x_size = len(x)
    phi_xy = np.zeros(x_size * y_size)
    start = y_candidate*x_size
    phi_xy[start:start+x_size] = x
    return phi_xy


def predict_word(perceptron, word_lines):
    w, b = perceptron
    y_hat = []
    prev_seq_phi = np.zeros(x_size * y_size)
    #since there is no features relating to order of the letters there is no significance
    # to use the viterbi algorithem, a greedy one at a time prediction will produce same results
    for line_i, line in enumerate(word_lines):
        phis = [phi_xi_yi(line[utils.PIXEL_VECTOR], y_candidate) for y_candidate in range(y_size)]
        results = [np.dot(w, phi_xy) + b[p_i] for p_i, phi_xy in enumerate(phis)]
        i_best = int(np.argmax(results))

        prev_seq_phi += phis[i_best]
        y_hat.append(i_best)

    return y_hat


def update_perceptron(perceptron, word_lines, y_hat_vec):
    w, b = perceptron
    for ex_i, example in enumerate(word_lines):
        x = np.array(example[utils.PIXEL_VECTOR])
        y = utils.L2I(example[utils.LETTER])
        y_hat = int(y_hat_vec[ex_i])
        if y != y_hat:
            phi_y = phi_xi_yi(x, y)
            phi_y_hat = phi_xi_yi(x, y_hat)
            w = w + (LR * (phi_y - phi_y_hat))
            b[y] = b[y] + LR
            b[y_hat] = b[y_hat] - LR

    return w, b


def train_perceptron(perceptron, train_data):
    for epoch in range(EPOCHS):
        print("epoch: %d started at %s" %(epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        good = bad = 0.0
        train_data = utils.word_shuffle(train_data)
        word_lines = []
        for line_i, line in enumerate(train_data):
            if line[utils.NEXT_ID] == -1:
                word_lines.append(line)
                y_hat = predict_word(perceptron, word_lines)
                w_good, w_bad = get_results(word_lines, y_hat)
                good += w_good
                bad += w_bad
                perceptron = update_perceptron(perceptron, word_lines, y_hat)
                word_lines = []
            else:
                word_lines.append(line)

        perc = (good / (good + bad)) * 100
        print("epoch no: %d acc = %.2f" % (epoch, perc))
    return perceptron


def test(perceptron, test_data):
    good = bad = 0.0
    word_lines = []
    for line_i, line in enumerate(test_data):
        if line[utils.NEXT_ID] == -1:
            word_lines.append(line)
            y_hat = predict_word(perceptron, word_lines)
            w_good, w_bad = get_results(word_lines, y_hat)
            good += w_good
            bad += w_bad
            word_lines = []
        else:
            word_lines.append(line)

    perc = (good / (good + bad)) * 100
    print("==========Test accuracy = %.2f==========" % perc)
    return perc


if __name__ == '__main__':
    train_input_file = "data/letters.train.data"
    test_input_file = "data/letters.test.data"
    if len(sys.argv) == 3:
        train_input_file = sys.argv[1]
        test_input_file = sys.argv[2]

    x_size = 8*16
    y_size = 26
    train_data = utils.load_data(train_input_file)
    perceptron = init_perceptron()
    perceptron = train_perceptron(perceptron, train_data)
    test_data = utils.load_data(test_input_file)
    test_acc = test(perceptron, test_data)
    print("finished at %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))