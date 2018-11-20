import numpy as np

from utils import init_param

init_lr = 0.3
reg_factor = 0.1

def set_learning_params(lr, regularization):
    global init_lr
    global reg_factor
    init_lr = lr
    reg_factor = regularization

def train_all_pairs(w, class_positive, class_negative, x, y, i):
    loss = None
    wt = w
    if y == class_positive or y == class_negative:
        yi = 1 if y == class_positive else -1
        d_lr = init_lr/((i+1)**0.5)
        loss = np.max([0, 1-yi*np.dot(w,x)])
        #  print("w: %d loss: %f" % (class_vs_all, loss))
        if (1-yi*np.dot(w,x)) >= 0:
            wt = (1-d_lr*reg_factor)*w + d_lr*yi*x
        else:
            wt = (1-d_lr*reg_factor) * w

    return (wt, loss)


def train(x_train, y_train):
    w01 = init_param(x_train[0])
    w02 = init_param(x_train[0])
    w03 = init_param(x_train[0])
    w12 = init_param(x_train[0])
    w13 = init_param(x_train[0])
    w23 = init_param(x_train[0])
    for i in range(len(y_train)):
        x = x_train[i]
        y = y_train[i]
        w01, w01loss = train_all_pairs(w01, 0, 1, x, y, i)
        w02, w02loss = train_all_pairs(w02, 0, 2, x, y, i)
        w03, w03loss = train_all_pairs(w03, 0, 3, x, y, i)
        w12, w12loss = train_all_pairs(w12, 1, 2, x, y, i)
        w13, w13loss = train_all_pairs(w13, 1, 3, x, y, i)
        w23, w23loss = train_all_pairs(w23, 2, 3, x, y, i)

    return (w01, w02, w03, w12, w13, w23)


def validate(param, x_dev, y_dev, dist_func):
    good = 0
    bad = 0
    (w01, w02, w03, w12, w13, w23) = param
    m = get_m()
    for i in range(len(y_dev)):
        x = x_dev[i]
        y = y_dev[i]
        y_hat = predict(m, (w01, w02, w03, w12, w13, w23), x, dist_func)
        if y_hat == y:
            good += 1
        else:
            bad += 1

    print ("precentage is: %f" % (good/(good+bad)))


def get_m():
    return np.matrix([[1, 1, 1, 0, 0, 0],
                      [-1, 0, 0, 1, 1, 0],
                      [0, -1, 0, -1, 0, 1],
                      [0, 0, -1, 0, -1, -1]])


def predict(m, weights, x, dist_func):
    (w01, w02, w03, w12, w13, w23) = weights
    fx01 = np.dot(w01, x)
    fx02 = np.dot(w02, x)
    fx03 = np.dot(w03, x)
    fx12 = np.dot(w12, x)
    fx13 = np.dot(w13, x)
    fx23 = np.dot(w23, x)
    fx_arr = [fx01, fx02, fx03, fx12, fx13, fx23]

    d0 = dist_func((m[0]).tolist()[0], fx_arr)
    d1 = dist_func((m[1]).tolist()[0], fx_arr)
    d2 = dist_func((m[2]).tolist()[0], fx_arr)
    d3 = dist_func((m[3]).tolist()[0], fx_arr)

    y_hat = np.argmin([d0, d1, d2, d3])
    return y_hat

def get_test_results(x_data, weights, dist_func):
    results = []
    for i in range(len(x_data)):
        x = x_data[i]
        m = get_m()
        y_hat = predict(m, weights, x, dist_func)
        results.append(y_hat)

    return results
