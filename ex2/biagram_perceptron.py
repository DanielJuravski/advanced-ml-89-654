from datetime import datetime
from random import shuffle

import utils
import numpy as np

EPOCHS = 1
LR = 1


def init_perceptron(x_size, y_size, vocab_size):
    eps = np.sqrt(6) / np.sqrt(x_size + y_size)
    size = (vocab_size * x_size) + (y_size * vocab_size)
    w = np.random.uniform(-eps, eps, size)
    b = np.random.uniform(-eps, eps, vocab_size)
    return (w,b)


def phi(x, y_cand, y_size, vocab_size, prev_y):
    x_size = len(x)
    size = (vocab_size * x_size) + (vocab_size * y_size)
    phi_xyprevy = np.zeros(size)
    x_start = y_cand*x_size
    phi_xyprevy[x_start:x_start+x_size] = x
    prevy_start = y_cand*vocab_size
    phi_xyprevy[prevy_start + prev_y] = 1
    return phi_xyprevy


def word_shuffle(train_data, shrink=-1):
    word = []
    all_words =[]
    for line in train_data:
        if line[utils.LETTER_POSITION] == 1:
            all_words.append(word)
            word = [line]
        else:
            word.append(line)

    shuffle(all_words)
    shuffled_lines = []
    if shrink == -1:
        shrink = len(all_words)
    for word in all_words[:shrink]:
        for line in word:
            shuffled_lines.append(line)

    return shuffled_lines


def train_perceptron(perceptron, data_by_index, y_size, vocab_size):
    train_data = list(data_by_index.values())
    w, b = perceptron
    for epoch in range(EPOCHS):
        print("epoch: %d started at %s" %(epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        good = bad = 0.0
        word_shuffle(train_data)
        for example in train_data:
            x = np.array(example[utils.PIXEL_VECTOR])
            y = utils.L2I(example[utils.LETTER])
            if example[utils.LETTER_POSITION] == 1:
                prev_y = utils.L2I('$')
            else:
                prev_example = data_by_index[example[utils.ID]-1]
                if prev_example[utils.NEXT_ID] == -1:
                    prev_y = utils.L2I('$')
                else:
                    prev_y = utils.L2I(prev_example[utils.LETTER])

            phis = [phi(x, y_cand, y_size, vocab_size, prev_y) for y_cand in range(vocab_size)]
            results = [np.dot(w, phi_xy)  for i, phi_xy in enumerate(phis)]
            y_hat = int(np.argmax(results))
            w = w + (LR * (phis[y] - phis[y_hat]))
            b[y] = b[y] + LR
            b[y_hat] = b[y_hat] - LR
            if y != y_hat:
                bad += 1
            else:
                good += 1

            iter = good+bad
            # if iter % 1000 == 0:
            #     print("iter: %d from %d" % (good+bad, len(train_data)))
            #print("y: %d yhat: %d" % (y,y_hat))

        perc = (good / (good + bad)) * 100
        print("epoch no: %d acc = %.2f" % (epoch, perc))
    return (w, b)


def test(perceptron, test_data, y_size, vocab_size):
    w, b = perceptron
    good = bad = 0.0
    for ex_i, example in enumerate(test_data):
        print("predicting line %d from %d" %(ex_i, len(test_data)))
        x = np.array(example[utils.PIXEL_VECTOR])
        y = utils.L2I(example[utils.LETTER])

        D_S = np.zeros((y_size,vocab_size))
        D_PI = np.zeros((y_size,vocab_size))

        prev_char = utils.L2I('$') #get scores for the first letter where the perv is $
        #initalization
        for i in range(0,vocab_size):
            curr_char = i+1
            phi_xy = phi(x, curr_char, y_size, vocab_size, prev_char)
            s = np.dot(w, phi_xy)
            D_S[0][i] = s
            D_PI[0][i] = 0
        #// ===================== RECURSION ===================== //
        for i in range(1, y_size):
            for j in range(vocab_size):
                curr_char = j+1
                phis = [phi(x, curr_char, y_size, vocab_size, prev_cand) for prev_cand in range(vocab_size)]
                results = [np.dot(w, phi_xy) + D_S[i-1][p_i] for p_i, phi_xy in enumerate(phis)]
                i_best = int(np.argmax(results))
                d_best = results[i_best]
                D_S[i][j] = d_best
                D_PI[i][j] = i_best

        #// ======================== BACK-TRACK ======================== //
        y_hat = np.zeros(y_size)
        d_best = -1
        for i in range(vocab_size):
            if d_best < D_S[y_size-1][i]:
                y_hat[y_size-1] = i
                d_best = D_S[y_size-1][i]

        for i in range(y_size-2, 0, -1):
            y_hat[i] = D_PI[i + 1][int(y_hat[i + 1])]

        pred = int(np.argmax(y_hat))
        print("pred=%d true=%d " %(pred, y))
        if y != pred:
            bad += 1
        else:
            good += 1

    perc = (good / (good + bad)) * 100
    print("==========Test accuracy = %.2f==========" % perc)
    return perc


def fill_data_by_index(train_data):
    by_idx = {}
    for example in train_data:
        by_idx[example[utils.ID]] = example

    return by_idx

if __name__ == '__main__':
    x_size = 8*16
    y_size = 26
    vocab_size = 27
    utils.ALPHABET = "$abcdefghijklmnopqrstuvwxyz"
    train_data = utils.load_data("data/letters.train.data")
    data_by_index = fill_data_by_index(train_data)

    perceptron = init_perceptron(x_size, y_size, vocab_size)
    perceptron = train_perceptron(perceptron, data_by_index, y_size, vocab_size)
    test_data = utils.load_data("data/letters.test.data")
    shrinked_test_data = word_shuffle(test_data, 5)
    test_acc = test(perceptron, shrinked_test_data, y_size, vocab_size)