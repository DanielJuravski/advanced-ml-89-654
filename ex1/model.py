import numpy as np

from utils import init_param


def set_learning_params(lr, regularization):
    global init_lr
    global reg_factor
    init_lr = lr
    reg_factor = regularization

def set_m(matrix):
    global m
    m = matrix

def set_dist_func(f):
    global dist_func
    dist_func = f

def train_custom(vector_dict, x, y, i):
    w = vector_dict['wi']
    if y in vector_dict['win']:
        y_binary = 1
    elif y in vector_dict['lose']:
        y_binary = -1
    else:
        return (w, None)

    d_lr = init_lr/((i+1)**0.5)
    loss = np.max([0, 1-y_binary*np.dot(w,x)])
    #print("classes:%r vs clasess:%r loss: %f" % (vector_dict['win'], vector_dict['lose'], loss))
    if (1-y_binary*np.dot(w,x)) >= 0:
        wt = (1-d_lr*reg_factor)*w + d_lr*y_binary*x
    else:
        wt = (1-d_lr*reg_factor) * w

    return (wt, loss)


def train(x_train, y_train):
    num_of_vectors = m.shape[1]
    vectors = []
    for i in range(num_of_vectors):
        vector = {}
        vector['i'] = i
        vector['wi'] = init_param(x_train[0])
        vector['win'], vector['lose'] = get_win_lose_classes_by_column(m, i)
        vectors.append(vector)

    for i in range(len(y_train)):
        x = x_train[i]
        y = y_train[i]
        for v in vectors:
            v['wi'], lossi = train_custom(v, x, y, i)

    weights = [v['wi'] for v in vectors]
    return weights


def validate(weights, x_dev, y_dev):
    good = 0
    bad = 0
    for i in range(len(y_dev)):
        x = x_dev[i]
        y = y_dev[i]
        y_hat = predict(weights, x)
        if y_hat == y:
            good += 1
        else:
            bad += 1

    print ("precentage is: %f" % (good/(good+bad)))


def predict(weights, x):
    fx_arr = []
    for w in weights:
        fx = np.dot(w, x)
        fx_arr.append(fx)

    d0 = dist_func((m[0]).tolist()[0], fx_arr)
    d1 = dist_func((m[1]).tolist()[0], fx_arr)
    d2 = dist_func((m[2]).tolist()[0], fx_arr)
    d3 = dist_func((m[3]).tolist()[0], fx_arr)

    y_hat = np.argmin([d0, d1, d2, d3])
    return y_hat

def get_test_results(x_data, weights):
    results = []
    for i in range(len(x_data)):
        x = x_data[i]
        y_hat = predict(weights, x)
        results.append(y_hat)

    return results

def get_win_lose_classes_by_column(m, col):
    num_of_classes = m.shape[0]
    win_classes = []
    lose_classes = []
    for i in range(num_of_classes):
        val = m[i].tolist()[0][col]
        if val == 1:
            win_classes.append(i)
        elif val == -1:
            lose_classes.append(i)
        else:
            continue

    return win_classes, lose_classes
