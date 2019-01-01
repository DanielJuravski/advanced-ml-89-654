from random import shuffle
import numpy as np

ID = 0
LETTER = 1
NEXT_ID = 2
FOLD = 3
PIXEL_VECTOR = 4
LETTER_POSITION = 5

ALPHABET = "abcdefghijklmnopqrstuvwxyz$"


def load_data(data_file):
    train_data = []
    data_by_index = {}
    with open(data_file) as f:
        content = f.readlines()
        for line in content:
            line_arr = line.split()
            id = int(line_arr[0])
            letter = line_arr[1]
            next_id = int(line_arr[2])
            letter_pos = int(line_arr[4])
            fold = line_arr[5]
            pixel_vector = line_arr[6:]
            pixel_vector = [int(x) for x in pixel_vector]
            data_by_index[id] = (id, letter, next_id, fold, pixel_vector, letter_pos)
            train_data.append((id, letter, next_id, fold, pixel_vector, letter_pos))


    return train_data

def I2L(index):
    return ALPHABET[index]

def L2I(letter):
    return ALPHABET.index(letter)


def word_shuffle(train_data, shrink=-1):
    word = []
    all_words = []
    for line in train_data:
        if line[LETTER_POSITION] == 1:
            all_words.append(word)
            word = [line]
        else:
            word.append(line)

    shuffle(all_words)
    shuffled_lines = []
    if shrink == -1:
        shrink = len(all_words)
    for word in all_words[:shrink]:
        for line in word:
            shuffled_lines.append(line)

    return shuffled_lines

def word_block_shuffle(train_data, shrink=-1):
    word = []
    word_block=[]
    all_words = []
    for line in train_data:
        if line[LETTER_POSITION] == 1:
            all_words.append(word)
            word = [line]
        else:
            word.append(line)

    shuffle(all_words)
    shuffled_lines = []
    if shrink == -1:
        shrink = len(all_words)
    for word in all_words[:shrink]:
        for line in word:
            shuffled_lines.append(line)

    return shuffled_lines

def get_results(word_lines, y_hat):
    good = bad = 0
    for i, line in enumerate(word_lines):
        y = L2I(line[LETTER])
        if y == y_hat[i]:
            good += 1
        else:
            bad += 1
    return good, bad


def print_results(all_words_pred, output_file):
    with open(output_file, "w") as f:
        for word in all_words_pred:
            for letter in word:
                f.write("%s\n" % I2L(int(letter)))

def print_results_by_letter(all_letters_pred, output_file):
    with open(output_file, "w") as f:
        for letter in all_letters_pred:
            f.write("%s\n" % I2L(int(letter)))