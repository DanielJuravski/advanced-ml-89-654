import numpy as np

from utils import init_param

init_lr =0.25
reg_factor = 0.05

def set_learning_params(lr, regularization):
    global init_lr
    global reg_factor
    init_lr = lr
    reg_factor = regularization


def train_tournament(w, win_classes, lose_classes, x, y, i):
    if y in win_classes:
        y_binary = 1
    elif y in lose_classes:
        y_binary = -1
    else:
        return (w, None)

    d_lr = init_lr/((i+1)**0.5)
    loss = np.max([0, 1-y_binary*np.dot(w,x)])
    #  print("w: %d loss: %f" % (class_vs_all, loss))
    if (1-y_binary*np.dot(w,x)) >= 0:
        wt = (1-d_lr*reg_factor)*w + d_lr*y_binary*x
    else:
        wt = (1-d_lr*reg_factor) * w

    return (wt, loss)


def train(x_train, y_train):
    w01v23 = init_param(x_train[0])
    w0v1 = init_param(x_train[0])
    w2v3 = init_param(x_train[0])
    for i in range(len(y_train)):
        x = x_train[i]
        y = y_train[i]
        w01v23, w0loss = train_tournament(w01v23, [0,1], [2,3], x, y, i)
        w0v1, w1loss = train_tournament(w0v1, [0], [1], x, y, i)
        w2v3, w2loss = train_tournament(w2v3, [2], [3], x, y, i)

    return w01v23, w0v1, w2v3

def validate(weights, x_dev, y_dev, dist_func):
    good = 0
    bad = 0
    m = get_m()
    for i in range(len(y_dev)):
        x = x_dev[i]
        y = y_dev[i]
        y_hat = predict(m, weights, x, dist_func)
        if y_hat == y:
            good += 1
        else:
            bad += 1

    print ("precentage is: %f" % (good/(good+bad)))


def get_m():
    return np.matrix([[1, 1, 0],
                      [1, -1, 0],
                      [-1, 0, 1],
                      [-1, 0, -1]])


def predict(m, weights, x, dist_func):
    (w01v23, w0v1, w2v3) = weights
    fx01v23 = np.dot(w01v23, x)
    fx0v1 = np.dot(w0v1, x)
    fx2v3 = np.dot(w2v3, x)
    fx_arr = [fx01v23, fx0v1, fx2v3]

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
