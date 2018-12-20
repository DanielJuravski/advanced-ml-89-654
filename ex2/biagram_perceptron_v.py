from datetime import datetime
import random
from random import shuffle

import utils
import numpy as np

random.seed(3)
np.random.seed(3)
EPOCHS = 3
LR = 0.1


def init_perceptron(x_size, y_size, vocab_size):
    eps = 0#np.sqrt(6) / np.sqrt(x_size + y_size)
    size = (y_size * x_size) + (vocab_size * y_size)
    w = np.random.uniform(-eps, eps, size)
    b = np.random.uniform(-eps, eps, y_size)
    return (w,b)


def phi(x, y_cand, y_size, vocab_size, prev_y):
    x_size = len(x)
    size = (y_size * x_size) + (vocab_size * y_size)
    phi_xyprevy = np.zeros(size)
    x_start = y_cand*x_size
    phi_xyprevy[x_start:x_start+x_size] = x

    prevy_start= (y_size * x_size) + (vocab_size * y_cand )
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


def update_perceptron(perceptron, word_lines, y_hat_vec, vocab_size, y_size):
    w, b = perceptron
    for ex_i, example in enumerate(word_lines):
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

        y_hat = int(y_hat_vec[ex_i])
        if y != y_hat:
            phi_y = phi(x, y, y_size, vocab_size, prev_y)
            phi_y_hat = phi(x, y_hat, y_size, vocab_size, prev_y)
            w = w + (LR * (phi_y - phi_y_hat))
            b[y] = b[y] + LR
            b[y_hat] = b[y_hat] - LR

    return (w, b)

def train_perceptron(perceptron, data_by_index, y_size, vocab_size):
    train_data = list(data_by_index.values())
    for epoch in range(EPOCHS):
        print("epoch: %d started at %s" %(epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        good = bad = 0.0
        train_data = word_shuffle(train_data)
        word_lines = []
        for line_i, line in enumerate(train_data):
            if line[utils.NEXT_ID] == -1:
                word_lines.append(line)
                w_good, w_bad, y_hat = predict_word(perceptron, word_lines, y_size, vocab_size)
                perceptron = update_perceptron(perceptron, word_lines, y_hat, vocab_size, y_size)
                good += w_good
                bad += w_bad
                word_lines = []
            else:
                word_lines.append(line)

        perc = (good / (good + bad)) * 100
        print("epoch no: %d acc = %.2f" % (epoch, perc))
    return perceptron


def predict_word(perceptron, word_lines, y_size, vocab_size, do_print=False):
    w, b = perceptron
    good = bad = 0.0
    y_word=""

    D_S = np.zeros((len(word_lines),vocab_size))
    D_PI = np.zeros((len(word_lines),vocab_size))

    #initalization
    prev_char = utils.L2I('$') #get scores for the first letter where the perv is $
    x = np.array(word_lines[0][utils.PIXEL_VECTOR])
    for i in range(y_size):
        curr_char = i
        phi_xy = phi(x, curr_char, y_size, vocab_size, prev_char)
        s = np.dot(w, phi_xy) + b[curr_char]
        D_S[0][i] = s
        D_PI[0][i] = 26
    #// ===================== RECURSION ===================== //
    for i in range(1, len(word_lines)):
        for j in range(y_size):
            curr_char = j
            phis = [phi(word_lines[i][utils.PIXEL_VECTOR], curr_char, y_size, vocab_size, prev_cand) for prev_cand in range(y_size)]
            results = [np.dot(w, phi_xy) + b[curr_char] + D_S[i-1][p_i] for p_i, phi_xy in enumerate(phis)]
            i_best = int(np.argmax(results))
            d_best = results[i_best]
            D_S[i][j] = d_best
            D_PI[i][j] = i_best

    #// ======================== BACK-TRACK ======================== //
    y_hat = np.zeros(len(word_lines))
    d_best = -1
    for i in range(y_size):
        if d_best < D_S[len(word_lines)-1][i]:
            y_hat[len(word_lines)-1] = i
            d_best = D_S[len(word_lines)-1][i]

    for i in range(len(word_lines)-2, -1, -1):
        y_hat[i] = D_PI[i + 1][int(y_hat[i + 1])]

    for i, line in enumerate(word_lines):
        pred = int(y_hat[i])
        y = utils.L2I(line[utils.LETTER])
        y_word +=line[utils.LETTER]
        if do_print:
            print("pred=%d true=%d " %(pred, y))
        if y != pred:
            bad += 1
        else:
            good += 1

    if do_print:
        print("word was %s first letter id is %d" % (y_word, word_lines[0][utils.ID]))
    return (good, bad, y_hat)


def test(perceptron, test_data, y_size, vocab_size):
    good = bad = 0.0
    word_lines = []
    for ex_i, line in enumerate(test_data):
        print("predicting line %d from %d" % (ex_i, len(word_lines)))
        if line[utils.NEXT_ID] == -1:
            word_lines.append(line)
            w_good, w_bad, y_hat = predict_word(perceptron, word_lines, y_size, vocab_size, True)
            good += w_good
            bad += w_bad
            word_lines = []
        else:
            word_lines.append(line)


    perc = (good / (good + bad)) * 100
    print("==========Test accuracy = %.2f==========" % perc)


def fill_data_by_index(train_data):
    by_idx = {}
    for example in train_data:
        by_idx[example[utils.ID]] = example

    return by_idx

if __name__ == '__main__':
    x_size = 8*16
    y_size = 26
    vocab_size = 27
    # utils.ALPHABET = "$abcdefghijklmnopqrstuvwxyz"
    train_data = utils.load_data("data/letters.train.data")
    data_by_index = fill_data_by_index(train_data)

    perceptron = init_perceptron(x_size, y_size, vocab_size)
    perceptron = train_perceptron(perceptron, data_by_index, y_size, vocab_size)
    test_data = utils.load_data("data/letters.test.data")
    # shrinked_test_data = word_shuffle(test_data)
    test_acc = test(perceptron, test_data, y_size, vocab_size)