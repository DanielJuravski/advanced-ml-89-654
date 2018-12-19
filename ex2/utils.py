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


