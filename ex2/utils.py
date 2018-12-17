ID = 0
LETTER = 1
NEXT_ID = 2
FOLD = 3
PIXEL_VECTOR = 4

ALPHABET = "abcdefghijklmnopqrstuvwxyz$"

def load_train(train_file):
    train_data = []
    with open(train_file) as f:
        content = f.readlines()
        for line in content:
            line_arr = line.split()
            id = line_arr[0]
            letter = line_arr[1]
            next_id = line_arr[2]
            fold = line_arr[5]
            pixel_vector = line_arr[6:]
            pixel_vector = [int(x) for x in pixel_vector]
            train_data.append((id, letter, next_id, fold, pixel_vector))

    return train_data

def I2L(index):
    return ALPHABET[index]

def L2I(letter):
    return ALPHABET.index(letter)


