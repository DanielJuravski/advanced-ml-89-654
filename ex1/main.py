import warnings

import numpy as np
from sklearn.utils import shuffle

import model
import utils


def calculate_dist_loss(m_row, fx_arr):
    def loss(a, b):
        return np.max([0, (1-(a * b))])
    points_loss = [loss(x[0], x[1]) for x in list(zip(m_row, fx_arr))]
    return np.sum(points_loss)


def calculate_dist_hamming(m_row, fx_arr):
    def compare_sign(a, b):
        return (1 - np.sign(a * b)) / 2
    differences = [compare_sign(x[0], x[1]) for x in list(zip(m_row, fx_arr))]
    return np.sum(differences)


def get_OA_m():
    return np.matrix([[ 1,-1,-1,-1],
                      [-1, 1,-1,-1],
                      [-1,-1, 1,-1],
                      [-1,-1,-1, 1]])


def get_AP_m():
    return np.matrix([[ 1, 1, 1, 0, 0, 0],
                      [-1, 0, 0, 1, 1, 0],
                      [ 0,-1, 0,-1, 0, 1],
                      [ 0, 0,-1, 0,-1,-1]])

def get_exahaustive_m():
    return np.matrix([[ 1, 1, 1, 1, 1, 1, 1],
                      [-1,-1,-1,-1, 1, 1, 1],
                      [-1,-1, 1, 1,-1,-1, 1],
                      [-1, 1,-1, 1,-1, 1,-1]])


def run_OA_hamming():
    model.set_learning_params(0.25, 0.05)
    model.set_m(get_OA_m())
    model.set_dist_func(calculate_dist_hamming)
    run("onevall.ham")


def run_AP_hamming():
    model.set_learning_params(0.5, 0.05)
    model.set_m(get_AP_m())
    model.set_dist_func(calculate_dist_hamming)
    run("allpairs.ham")


def run_custom_hamming():
    model.set_learning_params(0.5, 0.01)
    model.set_m(get_exahaustive_m())
    model.set_dist_func(calculate_dist_hamming)
    run("randm.ham")


def run_OA_loss():
    model.set_learning_params(0.25, 0.05)
    model.set_m(get_OA_m())
    model.set_dist_func(calculate_dist_loss)
    run("onevall.loss")


def run_AP_loss():
    model.set_learning_params(0.5, 0.05)
    model.set_m(get_AP_m())
    model.set_dist_func(calculate_dist_loss)
    run("allpairs.loss")


def run_custom_loss():
    model.set_learning_params(0.5, 0.01)
    model.set_m(get_exahaustive_m())
    model.set_dist_func(calculate_dist_loss)
    run("randm.loss")


def run(name):
    global x_train
    global y_train
    global x_dev
    global y_dev
    x_train, y_train = shuffle(x_train, y_train, random_state=1)
    x_dev, y_dev = shuffle(x_dev, y_dev, random_state=1)
    weights = model.train(x_train, y_train)
    print("validating: " + name)
    model.validate(weights, x_dev, y_dev)
    print("writing test predictions for " + name)
    results = model.get_test_results(x_test, weights)
    pred_path = "output/test." + name + ".pred"
    utils.print_results(results, pred_path)


if __name__ == '__main__':

    warnings.filterwarnings("ignore") # Ignore depricated warning from sklearn

    x_test = np.loadtxt("x4pred.txt")
    x_train, y_train, x_dev, y_dev = utils.load_data()
    run_OA_hamming()
    run_AP_hamming()
    run_custom_hamming()
    run_OA_loss()
    run_AP_loss()
    run_custom_loss()
