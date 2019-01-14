import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from time import gmtime, strftime


def get_line_data(n):
    # n number of samples to make
    x = np.random.randn(n, 1)
    y = x

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

        self.hidden0 = nn.Linear(n_features, 5)
        self.hidden1 = nn.Linear(5, 10)
        self.out = nn.Linear(10, n_out)
        self.f = torch.tanh

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
        line_data = get_line_data(batch_size)
        test_noise = get_rand_data(batch_size)
        real_data = Variable(line_data)
        # Generate fake data and detach (so gradients are not calculated for generator)
        fake_data = generator(test_noise).detach()
        N = real_data.size(0)
        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = discriminator(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real, ones_target(N))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake, zeros_target(N))
        error_fake.backward()

        # 1.3 Update weights with gradients
        optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer):
    # range beyond 1 works bad
    g_range = 1
    for i in range(g_range):
        test_noise = get_rand_data(batch_size)
        fake_data = generator(test_noise)
        N = fake_data.size(0)
        # Reset gradients
        optimizer.zero_grad()

        # Sample noise and generate fake data
        prediction = discriminator(fake_data)

        # Calculate error and backpropagate
        error = loss(prediction, ones_target(N))
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


if __name__ == '__main__':
    print("Start time: " + strftime("%H:%M:%S", gmtime()))
    batch_size = 50
    line_data = get_line_data(batch_size)
    #show_plot(line_data, batch_size)
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
    sgd_momentum = 0.9
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    loss = nn.BCELoss()

    num_epochs = 6000
    for epoch in range(num_epochs):
        # Train Discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer)

        # Train Generator
        g_error = train_generator(g_optimizer)

        # Display Progress
        if (epoch) % 100 == 0 and epoch > 1:
            print('Epoch: ' + str(epoch) +
                  ' D_err: ' + str(d_error.data.storage().tolist()[0]) +
                  #' D_pred_real: ' + str(d_pred_real) +
                  #' D_pred_fake: ' + str(d_pred_fake) +
                  ' G_err: ' + str(g_error.data.storage().tolist()[0]))
            pass
        if (epoch) % 1000 == 0 and epoch > 1:
            test_noise = get_rand_data(batch_size)
            fake_data = generator(test_noise)
            show_plot(fake_data, batch_size)
            pass


    print("End time: " + strftime("%H:%M:%S", gmtime()))