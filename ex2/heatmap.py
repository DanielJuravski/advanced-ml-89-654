import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import utils


def plot_heatmap(w):
    x_size = 8*16
    y_size = 26
    vocab_size = 27
    f_start = y_size*x_size
    matrix = np.empty((y_size,vocab_size))
    for i in range(y_size):
        line_letter = utils.ALPHABET[i]
        start = f_start + (i * vocab_size)
        vector = w[start:start+vocab_size]
        matrix[i] = np.array(vector)


    mmax, mmin = matrix.max(), matrix.min()
    matrix = (matrix - mmin)/(mmax - mmin)
    fig, ax = plt.subplots()
    # im = ax.pcolormesh(matrix, cmap=cm.gray, edgecolors='white', antialiased=True)
    im = ax.imshow(matrix, cmap=cm.gray_r)
    ax.set_xticks(np.arange(vocab_size))
    ax.set_yticks(np.arange(y_size))
    ax.set_xticklabels(utils.ALPHABET)
    plt.xlabel("previous letter")
    plt.ylabel("next letter")
    ax.set_yticklabels(utils.ALPHABET[:-1])
    ax.set_title("Charchters trasitions Parameters")
    fig.tight_layout()
    plt.savefig("w_heatmap.png")



if __name__ == '__main__':
    w = np.load("w_mat.npy")

    plot_heatmap(w)