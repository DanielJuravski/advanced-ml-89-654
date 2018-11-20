import numpy as np

from utils import init_param

init_lr =0.25
reg_factor = 0.05

def set_learning_params(lr, regularization):
    global init_lr
    global reg_factor
    init_lr = lr
    reg_factor = regularization


def train_w_vs_all(w, class_vs_all, x, y, i):
    if y == class_vs_all:
        y_binary = 1
    else:
        y_binary = -1

    d_lr = init_lr/((i+1)**0.5)
    loss = np.max([0, 1-y_binary*np.dot(w,x)])
  #  print("w: %d loss: %f" % (class_vs_all, loss))
    if (1-y_binary*np.dot(w,x)) >= 0:
        wt = (1-d_lr*reg_factor)*w + d_lr*y_binary*x
    else:
        wt = (1-d_lr*reg_factor) * w

    return (wt, loss)


def train(x_train, y_train):
    w0 = init_param(x_train[0])
    w1 = init_param(x_train[0])
    w2 = init_param(x_train[0])
    w3 = init_param(x_train[0])
    for i in range(len(y_train)):
        x = x_train[i]
        y = y_train[i]
        w0, w0loss = train_w_vs_all(w0, 0, x, y, i)
        w1, w1loss = train_w_vs_all(w1, 1, x, y, i)
        w2, w2loss = train_w_vs_all(w2, 2, x, y, i)
        w3, w3loss = train_w_vs_all(w3, 3, x, y, i)

    return (w0, w1, w2, w3)

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
    return np.identity(4)


def predict(m, weights, x, dist_func):
    (w0, w1, w2, w3) = weights
    fx0 = np.dot(w0, x)
    fx1 = np.dot(w1, x)
    fx2 = np.dot(w2, x)
    fx3 = np.dot(w3, x)
    fx_arr = [fx0, fx1, fx2, fx3]
    d0 = dist_func(m[0], fx_arr)
    d1 = dist_func(m[1], fx_arr)
    d2 = dist_func(m[2], fx_arr)
    d3 = dist_func(m[3], fx_arr)
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

