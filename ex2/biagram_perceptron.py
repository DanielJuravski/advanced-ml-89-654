#Notice this is the structured perceptron with biagram features
#This is the program that produces structured.pred

import random
import sys
from datetime import datetime

import numpy as np

import utils
from utils import word_shuffle

random.seed(3)
np.random.seed(3)
EPOCHS = 3
LR = 1

def init_perceptron():
    eps = 0#np.sqrt(6) / np.sqrt(x_size + y_size)
    size = (y_size * x_size) + (vocab_size * y_size)
    w = np.random.uniform(-eps, eps, size)
    b = np.random.uniform(-eps, eps, y_size)
    global w_accumelator, b_accumelator, t_counter
    w_accumelator = np.zeros(size)
    b_accumelator = np.zeros(y_size)
    t_counter = 0
    return (w,b)


def phi_xi_yi(x, y_cand, prev_y):
    x_size = len(x)
    size = (y_size * x_size) + (vocab_size * y_size)
    phi = np.zeros(size)
    x_start = y_cand*x_size
    phi[x_start:x_start+x_size] = x
    transition_i = int(prev_y) + (y_cand * vocab_size)
    index = (y_size * x_size) + transition_i
    phi[index] = 1.0
    return phi


def train_perceptron(perceptron, data_by_index):
    train_data = list(data_by_index.values())
    for epoch in range(EPOCHS):
        train_data = word_shuffle(train_data)
        print("epoch: %d started at %s" % (epoch, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        good = bad = 0.0
        word_lines = []
        for line_i, line in enumerate(train_data):
            if line[utils.NEXT_ID] == -1:
                word_lines.append(line)
                y_hat = predict_word(perceptron, word_lines)
                w_good, w_bad = utils.get_results(word_lines, y_hat)
                perceptron = update_perceptron(perceptron, word_lines, y_hat)
                good += w_good
                bad += w_bad
                word_lines = []
            else:
                word_lines.append(line)

        perc = (good / (good + bad)) * 100
        print("epoch no: %d acc = %.2f" % (epoch, perc))
    return perceptron


def predict_word(perceptron, word_lines, do_print=False):
    w, b = perceptron
    D_S = np.zeros((len(word_lines),y_size))
    D_PI = np.zeros((len(word_lines),y_size))
    #initalization
    prev_char = utils.L2I('$') #get scores for the first letter where the perv is $
    x = np.array(word_lines[0][utils.PIXEL_VECTOR])
    for i in range(y_size):
        curr_char = i
        phi_xy = phi_xi_yi(x, curr_char, prev_char)
        s = np.dot(w, phi_xy) + b[curr_char]
        D_S[0][i] = s
        D_PI[0][i] = utils.L2I('$')
    #// ===================== RECURSION ===================== //
    for i in range(1, len(word_lines)):
        line = word_lines[i]
        for curr_char in range(y_size):
            phis = [phi_xi_yi(line[utils.PIXEL_VECTOR], curr_char, prev_cand) for prev_cand in range(y_size)]
            results = [np.dot(w, phi_xy) + b[curr_char] + D_S[i-1][p_i] for p_i, phi_xy in enumerate(phis)]
            i_best = int(np.argmax(results))
            d_best = results[i_best]
            D_S[i][curr_char] = d_best
            D_PI[i][curr_char] = i_best

    #// ======================== BACK-TRACK ======================== //
    y_hat = np.zeros(len(word_lines))
    d_best = -1
    for last_char in range(y_size):
        if d_best < D_S[len(word_lines)-1][last_char]:
            y_hat[len(word_lines)-1] = last_char
            d_best = D_S[len(word_lines)-1][last_char]

    for word_i in range(len(word_lines)-2, -1, -1):
        y_hat[word_i] = D_PI[word_i + 1][int(y_hat[word_i + 1])]

    if do_print:
        y_word = ""
        pred_word = ""
        for i, line in enumerate(word_lines):
            pred = int(y_hat[i])
            y = utils.L2I(line[utils.LETTER])
            y_word +=line[utils.LETTER]
            pred_word +=utils.I2L(pred)
            print("pred=%d true=%d " %(pred, y))
        print("word: %s pred: %s first letter id is %d" % (y_word, pred_word, word_lines[0][utils.ID]))
    return y_hat


def update_perceptron(perceptron, word_lines, y_hat_vec):
    w, b = perceptron
    global w_accumelator, b_accumelator, t_counter
    t_counter += 1
    for ex_i, example in enumerate(word_lines):
        x = np.array(example[utils.PIXEL_VECTOR])
        y = utils.L2I(example[utils.LETTER])
        if ex_i == 0:
            prev_y = utils.L2I('$')
            prev_y_hat = utils.L2I('$')
        else:
            prev_example = data_by_index[example[utils.ID]-1]
            prev_y = utils.L2I(prev_example[utils.LETTER])
            prev_y_hat = y_hat_vec[ex_i -1]

        y_hat = int(y_hat_vec[ex_i])
        if y != y_hat:
            phi_y = phi_xi_yi(x, y, prev_y)
            phi_y_hat = phi_xi_yi(x, y_hat, prev_y_hat)
            w += (LR * (phi_y - phi_y_hat))
            b[y] += LR
            b[y_hat] -= LR

    w_accumelator += w
    b_accumelator += b
    return (w, b)

def test(perceptron, test_data):
    good = bad = 0.0
    word_lines = []
    all_words_resuls = []
    for ex_i, line in enumerate(test_data):
        # print("predicting line %d from %d" % (ex_i, len(word_lines)))
        if line[utils.NEXT_ID] == -1:
            word_lines.append(line)
            y_hat = predict_word(perceptron, word_lines, True)
            all_words_resuls.append(y_hat)
            w_good, w_bad = utils.get_results(word_lines, y_hat)
            good += w_good
            bad += w_bad
            word_lines = []
        else:
            word_lines.append(line)


    perc = (good / (good + bad)) * 100
    print("==========Test accuracy = %.2f==========" % perc)
    return all_words_resuls


def fill_data_by_index(train_data):
    by_idx = {}
    for example in train_data:
        by_idx[example[utils.ID]] = example

    return by_idx


def apply_average_update(w, w_accumelator, b, b_accumelator, t_counter):
    wdelta = w_accumelator / float(t_counter)
    bdelta = b_accumelator / float(t_counter)
    w += wdelta
    b += bdelta
    return w,b


if __name__ == '__main__':
    x_size = 8*16
    y_size = 26
    vocab_size = 27

    train_input_file = "data/letters.train.data"
    test_input_file = "data/letters.test.data"
    output_file = "structured.pred"
    if len(sys.argv) == 4:
        train_input_file = sys.argv[1]
        test_input_file = sys.argv[2]
        output_file = sys.argv[3]

    train_data = utils.load_data(train_input_file)
    data_by_index = fill_data_by_index(train_data)

    w_accumelator = None
    b_accumelator = None
    t_counter = 0

    perceptron = init_perceptron()
    w,b = perceptron
    w0 = np.copy(w)
    b0 = np.copy(b)
    perceptron = train_perceptron(perceptron, data_by_index)
    wf, bf = apply_average_update(w0, w_accumelator, b0, b_accumelator, t_counter)
    np.save("w_mat", wf)
    np.save("b_vec", bf)

    test_data = utils.load_data(test_input_file)
    all_words_pred = test((wf,bf), test_data)
    utils.print_results(all_words_pred, output_file)
    print("finished at %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))