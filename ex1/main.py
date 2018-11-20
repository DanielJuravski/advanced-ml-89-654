import numpy as np

from utils import load_data, calculate_dist_hamming, calculate_dist_loss, print_results


def run_OA_hamming(test_x):
    from OA import set_learning_params, train, validate, get_test_results
    set_learning_params(0.25, 0.05)
    run(test_x, train, validate, get_test_results, calculate_dist_hamming, "onevall.ham")


def run_AP_hamming(test_x):
    from AP import set_learning_params, train, validate, get_test_results
    set_learning_params(0.5, 0.05)
    run(test_x, train, validate, get_test_results, calculate_dist_hamming, "allpairs.ham")


def run_tournament_hamming(test_x):
    from tournament import set_learning_params, train, validate, get_test_results
    set_learning_params(0.35, 0.05)
    run(test_x, train, validate, get_test_results, calculate_dist_hamming, "randm.ham")


def run_OA_loss(test_x):
    from OA import set_learning_params, train, validate, get_test_results
    set_learning_params(0.25, 0.05)
    run(test_x, train, validate, get_test_results, calculate_dist_loss, "onevall.loss")


def run_AP_loss(test_x):
    from AP import set_learning_params, train, validate, get_test_results
    set_learning_params(0.5, 0.05)
    run(test_x, train, validate, get_test_results, calculate_dist_loss, "allpairs.loss")


def run_tournament_loss(test_x):
    from tournament import set_learning_params, train, validate, get_test_results
    set_learning_params(0.35, 0.05)
    run(test_x, train, validate, get_test_results, calculate_dist_loss, "randm.loss")


def run(test_x, train_func, validate_func, get_results_func, calc_dist_func, name):
    x_train, y_train, x_dev, y_dev = load_data()
    weights = train_func(x_train, y_train)
    print("validating: " + name)
    validate_func(weights, x_dev, y_dev, calc_dist_func)
    print("writing test predictions for " + name)
    results = get_results_func(test_x, weights, calc_dist_func)
    pred_path = "output/test." + name + ".pred"
    print_results(results, pred_path)



if __name__ == '__main__':
    x_test = np.loadtxt("x_test.txt")

    run_OA_hamming(x_test)
    run_AP_hamming(x_test)
    run_tournament_hamming(x_test)
    run_OA_loss(x_test)
    run_AP_loss(x_test)
    run_tournament_loss(x_test)
