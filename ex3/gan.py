import os
import sys
from time import gmtime, strftime, time

import numpy as np
import torch

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd.variable import Variable
import random

torch.manual_seed(66)
np.random.seed(66)


EPOCHS_TILL_PRINT = 1000


# added random value for multiple runs
prefix = ""

PICS_DIR = prefix +"./statistics_" + str(time()) + "/"

def get_line_data(n):
    # n number of samples to make
    x = np.random.randn(n, 1)
    y = x

    x /= x.max()
    y /= y.max()

    data = np.hstack((x, y))
    data_tensor = torch.Tensor(data)

    return data_tensor


def get_parabola_data(n):
    # n number of samples to make
    x = np.random.randn(n, 1)
    y = x**2

    x /= x.max()
    y /= y.max()

    data = np.hstack((x, y))
    data_tensor = torch.Tensor(data)

    return data_tensor


def get_spiral_data(n_points):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    x = -np.cos(n) * n + np.random.rand(n_points, 1)
    y = np.sin(n) * n + np.random.rand(n_points, 1)

    x /= x.max()
    y /= y.max()

    data = np.hstack((x, y))
    data_tensor = torch.Tensor(data)

    return data_tensor


def get_rand_data(n):
    # n number of samples to make
    x = np.random.uniform(size=(n, 1))
    y = np.random.uniform(size=(n, 1))
    #x = np.random.randn(n, 1)
    #y = np.random.randn(n, 1)

    #x /= x.max()
    #y /= y.max()

    data = np.hstack((x, y))
    data_tensor = Variable(torch.Tensor(data))

    return data_tensor


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 2
        n_out = 1

        self.hidden0 = nn.Linear(n_features, DL1_SIZE)
        nn.init.normal_(self.hidden0.weight.data, mean=0.0, std=0.02)
        self.hidden1 = nn.Linear(DL1_SIZE, DL2_SIZE)
        nn.init.normal_(self.hidden1.weight.data, mean=0.0, std=0.02)
        self.hidden2 = nn.Linear(DL2_SIZE, DL3_SIZE)
        nn.init.normal_(self.hidden2.weight.data, mean=0.0, std=0.02)
        self.out = nn.Linear(DL3_SIZE, n_out)
        nn.init.normal_(self.out.weight.data, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(D_REGULARIZATION_RATE)
        self.bn1 = nn.BatchNorm1d(DL1_SIZE)
        self.bn2 = nn.BatchNorm1d(DL2_SIZE)
        self.bn3 = nn.BatchNorm1d(DL3_SIZE)
        self.f1 = get_activation_func(D_ACTIVATION_FUNC_1)
        self.f2 = get_activation_func(D_ACTIVATION_FUNC_2)


    def forward(self, x):
        if REG_TYPE == 'dropout':
            x = self.dropout(self.f1(self.bn1(self.hidden0(x))))
            x = self.dropout(self.f1(self.bn2(self.hidden1(x))))
            x = self.dropout(self.f1(self.bn3(self.hidden2(x))))
            x = self.f2(self.out(x))
        else:
            x = self.bn1(self.f1(self.hidden0(x)))
            x = self.bn2(self.f1(self.hidden1(x)))
            x = self.bn3(self.f1(self.hidden2(x)))
            x = self.f2(self.out(x))

        return x


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 2
        n_out = 2

        self.hidden0 = nn.Linear(n_features, GL1_SIZE)
        nn.init.normal_(self.hidden0.weight.data, mean=0.0, std=0.02)
        self.hidden1 = nn.Linear(GL1_SIZE, GL2_SIZE)
        nn.init.normal_(self.hidden1.weight.data, mean=0.0, std=0.02)
        self.hidden2 = nn.Linear(GL2_SIZE, GL3_SIZE)
        nn.init.normal_(self.hidden2.weight.data, mean=0.0, std=0.02)
        self.bn1 = nn.BatchNorm1d(GL1_SIZE)
        self.bn2 = nn.BatchNorm1d(GL2_SIZE)
        self.bn3 = nn.BatchNorm1d(GL3_SIZE)
        self.out = nn.Linear(GL3_SIZE, n_out)
        nn.init.normal_(self.out.weight.data, mean=0.0, std=0.02)
        self.f1 = get_activation_func(G_ACTIVATION_FUNC_1)
        self.f2 = get_activation_func(G_ACTIVATION_FUNC_2)
        self.f3 = get_activation_func(G_ACTIVATION_FUNC_3)
        self.dropout = nn.Dropout(G_REGULARIZATION_RATE)



    def forward(self, x):
        if REG_TYPE == 'dropout':
            x = self.dropout(self.f1(self.bn1(self.hidden0(x))))
            x = self.dropout(self.f1(self.bn2(self.hidden1(x))))
            x = self.dropout(self.f1(self.bn3(self.hidden2(x))))
            x = self.out(x)
        else:
            x = self.f1(self.hidden0(x))
            x = self.f1(self.hidden1(x))
            x = self.f1(self.hidden2(x))
            x = self.out(x)

        return x


def ones_target(size, axs=2):
    '''
    Tensor containing ones, with shape = size
    '''
    eps = EPS
    if EPS == "dynamic":
        eps = (random.random() * 0.5) - 0.2
    data = Variable(torch.ones(size, axs))
    return torch.full((size,), 1)


def zeros_target(size, axs=2):
    '''
    Tensor containing zeros, with shape = size
    '''
    eps = EPS
    if EPS == "dynamic":
        eps = (random.random() * 0.2)
    data = Variable(torch.zeros(size, axs))
    return torch.full((size, ), 0)


def train_discriminator(optimizer):
    d_range = 1
    for i in range(d_range):
        # Reset gradients
        optimizer.zero_grad()

        sample_data = get_real_data_func(DATA_TYPE)
        real_data = Variable(sample_data(BATCH_SIZE))
        noise = get_rand_data(BATCH_SIZE)
        # Generate fake data and detach (so gradients are not calculated for generator)
        fake_data = generator(noise).detach()



        # 1.1 Train on Real Data
        prediction_real = discriminator(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real.view(-1), ones_target(BATCH_SIZE, 1))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake.view(-1), zeros_target(BATCH_SIZE, 1))
        error_fake.backward()

        # 1.3 Update weights with gradients
        optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer):
    global FLIP_GENERATOR
    # range beyond 1 works bad
    g_range = 1
    for i in range(g_range):
        optimizer.zero_grad()

        noise = get_rand_data(BATCH_SIZE)
        # Reset gradients


        fake_data = generator(noise)
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)

        # Calculate error and backpropagate
        if FLIP_GENERATOR:
            target2 = zeros_target(BATCH_SIZE)
            error2 = -loss(prediction, target2)
            error2.backward(retain_graph=True)

        target1 = ones_target(BATCH_SIZE)
        error1 = loss(prediction.view(-1), target1)
        error1.backward()

        # Update weights with gradients
        optimizer.step()

    # Return error
    return (error1 + error2) if FLIP_GENERATOR else error1


def show_plot(data, batch_size, suffix):
    out_path = PICS_DIR+"output_"+str(suffix)
    data = data.data.storage().tolist()
    X_hat = np.array([[data[0::2][i]] for i in range(batch_size)])
    Y_hat = np.array([[data[1::2][i]] for i in range(batch_size)])

    data_con = np.hstack((X_hat, Y_hat))
    fig = plt.figure()
    plt.plot(data_con[:, 0], data_con[:, 1], '.')
    #plt.title('training set')
    plt.savefig(out_path)
    plt.close(fig)
    # plt.show()


def clean_dir(path):
    import glob

    files = glob.glob(path+"*")
    for f in files:
        os.remove(f)


def get_real_data_func(data_type):
    if data_type == 'line':
        real_data_fun = get_line_data
    elif data_type == 'parabola':
        real_data_fun = get_parabola_data
    elif data_type == 'spiral' or data_type == 'spirala':
        real_data_fun = get_spiral_data
    else:
        raise Exception('Wrong data type')

    return real_data_fun

def get_activation_func(f_name):
    if f_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif f_name == "tanh":
        return torch.nn.Tanh()
    elif f_name == "relu":
        return torch.nn.ReLU(True)
    elif f_name == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif f_name == "soft_sign":
        return torch.nn.Softsign()
    elif f_name == "celu":
        return torch.nn.CELU()
    elif f_name == "tanh_shrink":
        return torch.nn.Tanhshrink()
    elif f_name == "soft_shrink":
        return torch.nn.Softshrink()



def print_params():
    with open(PICS_DIR + 'params.txt', 'w') as f:
        f.write("data type: %s\n"
                "batch size: %f\n"
                "epochs: %f\n"
                "g reg rate: %f\n"
                "d reg rate: %f\n"
                "reg type: %s\n"
                "g activation_1 %s\n"
                "g activation_2 %s\n"
                "g activation 3 %s\n"
                "d activation_1 %s\n"
                "d activation_2 %s\n"
                "d lr %f\n"
                "g lr %f\n"
                "2step on G\n"
                "Epsilon %s\n"
                "raised neurons to 10,20\n"
                "added layer to d and g\n"
                "added detach on g with allow_unreachable=True"
                # "added bn on 2 layers"
                % (DATA_TYPE, BATCH_SIZE, EPOCHS, G_REGULARIZATION_RATE, D_REGULARIZATION_RATE, REG_TYPE,
                   G_ACTIVATION_FUNC_1, G_ACTIVATION_FUNC_2, G_ACTIVATION_FUNC_3, D_ACTIVATION_FUNC_1, D_ACTIVATION_FUNC_2, D_LR, G_LR, str(EPS)))


if __name__ == '__main__':
    print("Start time: " + strftime("%H:%M:%S", gmtime()))

    if len(sys.argv) > 1:
        DATA_TYPE = sys.argv[1]
    else:
        DATA_TYPE = 'spiral'

    G_REGULARIZATION_RATE = 0
    D_REGULARIZATION_RATE = 0
    BATCH_SIZE_TARGET = 1000

    if DATA_TYPE != 'spiral':
        FLIP_GENERATOR = False

    if DATA_TYPE == "line":
        DL1_SIZE = 20
        DL2_SIZE = 30
        DL3_SIZE = 20
        GL1_SIZE = 20
        GL2_SIZE = 30
        GL3_SIZE = 20
        FLIP_GENERATOR = False
        D_ACTIVATION_FUNC_1 = "relu"
        D_ACTIVATION_FUNC_2 = "sigmoid"
        G_ACTIVATION_FUNC_1 = "relu"
        G_ACTIVATION_FUNC_2 = "tanh"
        G_ACTIVATION_FUNC_3 = "tanh_shrink"
        REG_TYPE = 'dropout'
        G_LR = 0.0003
        D_LR = 0.0001
        BATCH_SIZE = 1000
        EPOCHS = 40 * 1000
        EPS = 0.01
    elif DATA_TYPE == "parabola":
        DL1_SIZE = 20
        DL2_SIZE = 30
        DL3_SIZE = 20
        GL1_SIZE = 20
        GL2_SIZE = 30
        GL3_SIZE = 20
        FLIP_GENERATOR = False
        D_ACTIVATION_FUNC_1 = "relu"
        D_ACTIVATION_FUNC_2 = "sigmoid"
        G_ACTIVATION_FUNC_1 = "relu"
        G_ACTIVATION_FUNC_2 = "tanh"
        G_ACTIVATION_FUNC_3 = "tanh_shrink"
        REG_TYPE = 'dropout'
        G_LR = 0.0003
        D_LR = 0.0001
        BATCH_SIZE = 1000
        EPOCHS = 80 * 1000
        EPS = 0
    elif DATA_TYPE == 'spiral':
        DL1_SIZE = 20
        DL2_SIZE = 30
        DL3_SIZE = 20
        GL1_SIZE = 20
        GL2_SIZE = 30
        GL3_SIZE = 20
        FLIP_GENERATOR = False
        D_ACTIVATION_FUNC_1 = "relu"
        D_ACTIVATION_FUNC_2 = "sigmoid"
        G_ACTIVATION_FUNC_1 = "relu"
        G_ACTIVATION_FUNC_2 = "tanh"
        G_ACTIVATION_FUNC_3 = "tanh_shrink"
        REG_TYPE = 'dropout'
        G_LR = 0.0003
        D_LR = 0.0001
        BATCH_SIZE = 1000
        EPOCHS =100 * 1000
        EPS = 0

    clean_dir(PICS_DIR)
    if not os.path.exists(PICS_DIR):
        os.makedirs(PICS_DIR)

    #target data
    sample_data = get_real_data_func(DATA_TYPE)
    real_data = Variable(sample_data(BATCH_SIZE))
    show_plot(real_data, BATCH_SIZE, "target_data")

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    if REG_TYPE == 'weight_decay':
        d_optimizer = optim.Adam(discriminator.parameters(), lr=D_LR, weight_decay=D_REGULARIZATION_RATE)
        g_optimizer = optim.Adam(generator.parameters(), lr=G_LR, weight_decay=G_REGULARIZATION_RATE)
    else:
        d_optimizer = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    loss = nn.BCELoss()

    print_params()
    generator.train()
    discriminator.train()
    for epoch in range(EPOCHS):

        # Train Discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer)

        # Train Generator
        g_error = train_generator(g_optimizer)

        # Display Progress
        if (epoch) % 100 == 0 and epoch > 1:
            print('Epoch: ' + str(epoch) +
                  ' D_err: ' + str(d_error.data.storage().tolist()[0]) +
                  ' G_err: ' + str(g_error.data.storage().tolist()[0]))


        if (epoch) % EPOCHS_TILL_PRINT == 0 and epoch > 1:
            test_noise = get_rand_data(1000)
            fake_data = generator(test_noise)
            show_plot(fake_data, 1000, int(epoch/EPOCHS_TILL_PRINT))
            pass

    # plot final plot
    generator.eval()
    discriminator.eval()
    test_noise = get_rand_data(BATCH_SIZE_TARGET)
    fake_data = generator(test_noise)
    show_plot(fake_data, BATCH_SIZE_TARGET, "final")
    # write parameters tunings
    print("End time: " + strftime("%H:%M:%S", gmtime()))



