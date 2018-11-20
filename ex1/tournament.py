from utils import init_param
import numpy as np

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


