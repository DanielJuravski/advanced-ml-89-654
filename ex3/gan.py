import os
import sys
from time import gmtime, strftime, time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd.variable import Variable
import random

FLIP_GENERATOR_BACK_EPOCH = 30000

FLIP_GENERATOR = True

EPOCHS_TILL_PRINT = 1000

# added random value for multiple runs
PICS_DIR = "./statistics_" + str(time()) + "/"

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
        n_out = 2

        self.hidden0 = nn.Linear(n_features, 5)
        self.hidden1 = nn.Linear(5, 10)
        self.out = nn.Linear(10, n_out)
        self.f = torch.sigmoid


    def forward(self, x):
        x = self.f(self.hidden0(x))
        x = self.f(self.hidden1(x))
        x = self.f(self.out(x))

        return x


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 2
        n_out = 2

        self.hidden0 = nn.Linear(n_features, 6)
        self.hidden1 = nn.Linear(6, 10)
        self.out = nn.Linear(10, n_out)
        self.f = torch.tanh
        self.regularization = nn.Dropout(G_REGULARIZATION_RATE)


    def forward(self, x):
        if REG_TYPE == 'dropout':
            x = self.f(self.regularization(self.hidden0(x)))
            x = self.f(self.regularization(self.hidden1(x)))
            x = self.f(self.out(x))
        else:
            x = self.f(self.hidden0(x))
            x = self.f(self.hidden1(x))
            x = self.f(self.out(x))

        return x


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 2)) -0.01
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 2)) + 0.01
    return data


def train_discriminator(optimizer):
    d_range = 20
    for i in range(d_range):
        sample_data = get_real_data_func(DATA_TYPE)
        real_data = Variable(sample_data(BATCH_SIZE))
        noise = get_rand_data(BATCH_SIZE)
        # Generate fake data and detach (so gradients are not calculated for generator)
        fake_data = generator(noise).detach()

        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = discriminator(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real, ones_target(BATCH_SIZE))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake, zeros_target(BATCH_SIZE))
        error_fake.backward()

        # 1.3 Update weights with gradients
        optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer):
    # range beyond 1 works bad
    g_range = 1
    for i in range(g_range):
        noise = get_rand_data(BATCH_SIZE)


        # Reset gradients
        optimizer.zero_grad()

        fake_data = generator(noise)
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)

        # Calculate error and backpropagate
        target = ones_target(BATCH_SIZE) if not FLIP_GENERATOR else zeros_target(BATCH_SIZE)
        factor = 1 if not FLIP_GENERATOR else -1
        error = loss(prediction, target) * factor
        error.backward()

        # Update weights with gradients
        optimizer.step()

    # Return error
    return error


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


if __name__ == '__main__':
    print("Start time: " + strftime("%H:%M:%S", gmtime()))

    if len(sys.argv) > 5:
        DATA_TYPE = sys.argv[1]
        BATCH_SIZE = int(sys.argv[2])
        EPOCHS = int(sys.argv[3])
        G_REGULARIZATION_RATE = float(sys.argv[4])
        D_REGULARIZATION_RATE = float(sys.argv[5])
        REG_TYPE = sys.argv[6]

    else:
        DATA_TYPE = 'spiral'
        BATCH_SIZE = 2000
        EPOCHS = 100000
        G_REGULARIZATION_RATE = 0.3
        D_REGULARIZATION_RATE = 0.0
        REG_TYPE = 'none'

    BATCH_SIZE_TARGET = 1000

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
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, weight_decay=D_REGULARIZATION_RATE)
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, weight_decay=G_REGULARIZATION_RATE)
    else:
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    loss = nn.BCELoss()

    for epoch in range(EPOCHS):
        if epoch > 70000:
            FLIP_GENERATOR = False
        elif (epoch-1) > FLIP_GENERATOR_BACK_EPOCH:
            if epoch % 10000 == 0:
                FLIP_GENERATOR = not FLIP_GENERATOR

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
    test_noise = get_rand_data(BATCH_SIZE_TARGET)
    fake_data = generator(test_noise)
    show_plot(fake_data, BATCH_SIZE_TARGET, "final")
    # write parameters tunings
    with open(PICS_DIR + 'params.txt', 'w') as f:
        f.write("data type: %s\n"
                "batch size: %f\n"
                "epochs: %f\n"
                "g reg rate: %f\n"
                "d reg rate: %f\n"
                "reg type: %s\n"
                % (DATA_TYPE, BATCH_SIZE, EPOCHS, G_REGULARIZATION_RATE, D_REGULARIZATION_RATE, REG_TYPE))
    print("End time: " + strftime("%H:%M:%S", gmtime()))



