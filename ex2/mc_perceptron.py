import utils
import numpy as np
from random import shuffle

EPOCHS = 70
LR = 0.0001

def init_perceptron(x_size, y_size):
    eps = np.sqrt(6) / np.sqrt(x_size + y_size)
    w = np.random.uniform(-eps, eps, (y_size, x_size))
    b = np.random.uniform(-eps, eps, y_size)
    return (w, b)


def get_eta(i, y, y_hat):
    if i == y:
        return 1
    elif i == y_hat:
        return -1
    else:
        return 0


def train_perceptron(perceptron, train_data):
    w,b = perceptron
    for epoch in range(EPOCHS):
        good = bad = 0.0
        shuffle(train_data)
        for example in train_data:
            x = np.array(example[utils.PIXEL_VECTOR])
            y = utils.L2I(example[utils.LETTER])
            mul = np.dot(w, x) + b
            y_hat = np.argmax(mul)
            if y != y_hat:
                bad += 1
                # eta_vec = np.zeros(26)
                # eta_vec[y] = LR
                # eta_vec[y_hat] = -LR
                # w = w + (np.dot(np.array(eta_vec), x))
                for i, row in enumerate(w):
                    if i == y:
                        w[i] = row + LR * x
                        b[i] += LR
                    elif i==y_hat:
                        w[i] = row - LR * x
                        b[i] -= LR
            else:
                good += 1

        perc = (good / (good + bad)) * 100
        print("epoch no: %d acc = %.2f" % (epoch, perc))
    return (w, b)


def test(perceptron, test_data):
    w,b = perceptron
    good = bad = 0.0
    for example in test_data:
        x = np.array(example[utils.PIXEL_VECTOR])
        y = utils.L2I(example[utils.LETTER])
        mul = np.dot(w, x) + b
        y_hat = np.argmax(mul)
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
    train_data = utils.load_train("data/letters.train.data")
    perceptron = init_perceptron(x_size, y_size)
    perceptron = train_perceptron(perceptron, train_data)
    test_data = utils.load_train("data/letters.test.data")
    test_acc = test(perceptron, test_data)
