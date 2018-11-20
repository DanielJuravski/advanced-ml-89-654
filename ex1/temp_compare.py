import numpy as np
if __name__ == '__main__':
    test_file = "test.randm.ham.pred"
    y_test = np.loadtxt("y_test.txt")
    y_pred = np.loadtxt("output/"+test_file)
    good = 0
    bad = 0
    for i in range(len(y_pred)):
        pred=y_pred[i]
        act=int(y_test[i])
        if pred==act:
            good+=1
        else:
            bad+=1

    print("prediction for " + test_file + " is: %f" % (good/(good+bad)))
