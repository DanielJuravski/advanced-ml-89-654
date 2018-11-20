from utils import load_data, init_param
import numpy as np

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



