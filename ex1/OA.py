from .utils import load_data, init_param
import numpy as np

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



