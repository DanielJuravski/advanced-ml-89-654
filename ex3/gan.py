import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from time import gmtime, strftime
import sys


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


def get_spiral_data(n):
    pass


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

        self.hidden0 = nn.Linear(n_features, 5)
        self.hidden1 = nn.Linear(5, 10)
        self.out = nn.Linear(10, n_out)
        self.f = torch.tanh
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.f(self.hidden0(x))
        x = self.f(self.hidden1(x))
        x = self.f(self.out(x))

        return x


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 2))
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 2))
    return data


def train_discriminator(optimizer):
    d_range = 20
    for i in range(d_range):
        sample_data = get_real_data(DATA_TYPE)
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
        fake_data = generator(noise)

        # Reset gradients
        optimizer.zero_grad()

        # Sample noise and generate fake data
        prediction = discriminator(fake_data)

        # Calculate error and backpropagate
        error = loss(prediction, ones_target(BATCH_SIZE))
        error.backward()

        # Update weights with gradients
        optimizer.step()

    # Return error
    return error


def show_plot(data, batch_size):
    data = data.data.storage().tolist()
    X_hat = np.array([[data[0::2][i]] for i in range(batch_size)])
    Y_hat = np.array([[data[1::2][i]] for i in range(batch_size)])

    data_con = np.hstack((X_hat, Y_hat))

    plt.plot(data_con[:, 0], data_con[:, 1], '.')
    plt.title('training set')
    plt.savefig('output')
    plt.show()


def get_real_data(data_type):
    if data_type == 'line':
        real_data_fun = get_line_data
    elif data_type == 'parabola':
        real_data_fun = get_parabola_data
    elif data_type == 'spiral':
        real_data_fun = get_spiral_data
    else:
        raise Exception('Wrong data type')

    return real_data_fun


if __name__ == '__main__':
    print("Start time: " + strftime("%H:%M:%S", gmtime()))

    if len(sys.argv) > 2:
        DATA_TYPE = sys.argv[1]
        BATCH_SIZE = int(sys.argv[2])
        EPOCHS = int(sys.argv[3])
    else:
        DATA_TYPE = 'parabola'
        BATCH_SIZE = 20
        EPOCHS = 40000

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    loss = nn.BCELoss()

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

    test_noise = get_rand_data(500)
    fake_data = generator(test_noise)
    show_plot(fake_data, 500)
    print("End time: " + strftime("%H:%M:%S", gmtime()))