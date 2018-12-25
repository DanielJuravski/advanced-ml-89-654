import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import utils


def plot_heatmap(w):
    x_size = 8*16
    y_size = 26
    vocab_size = 27
    f_start = y_size*x_size
    matrix = np.empty((vocab_size, y_size))
    for prev in range(vocab_size):
        for curr in range(y_size):
            start = f_start + (curr * vocab_size)
            val = w[start+prev]
            matrix[prev][curr] = val


    mmax, mmin = matrix.max(), matrix.min()
    matrix = (matrix - mmin)/(mmax - mmin)
    fig, ax = plt.subplots()
    # im = ax.pcolormesh(matrix, cmap=cm.gray, edgecolors='white', antialiased=True)
    im = ax.imshow(matrix, cmap=cm.gray_r)
    ax.set_xticks(np.arange(y_size))
    ax.set_yticks(np.arange(vocab_size))
    ax.set_xticklabels(utils.ALPHABET[:-1])
    plt.xlabel("next letter")
    plt.ylabel("previous letter")
    ax.set_yticklabels(utils.ALPHABET)
    ax.set_title("Charchters trasitions Parameters")
    fig.tight_layout()
    plt.savefig("w_heatmap.png")



if __name__ == '__main__':
    w = np.load("w_mat.npy")

    plot_heatmap(w)