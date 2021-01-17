# Recurrent Neural Network - what we use for sequences
# use overview column as features

import tensorflow as tf
import numpy as np
import pandas as pd


def train_RNN(df):
    x_train, y_train = arrange(df)
    for row in range(x_train.shape[0]):
        features = get_features(x_train, row)
        label = y_train.at[row, 'popularity']


def arrange(df):
    df.dropna(0, subset=['overview'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    x_train = df[['overview']]
    y_train = df[['popularity']]
    return x_train, y_train


def get_features(x_train, row):
    text = x_train.at[row, 'overview']
    words = np.array(text.split())  # create an array of all the words in the overview
    unique_words = np.unique(words)  # an array of unique words, to use as features
    encoding_mat = pd.get_dummies(unique_words)
    features = []
    for word in words:
        encode = list(encoding_mat.loc[:, word])
        features.append(encode)  # add the one-hot-encoding vector that represents the word as feature
    return features

"""
def amos(features, batch_size):
    num_input_examples = len(features)
    curr_batch = 0

    def next_batch():  # not that efficient. Can't save everything in memory, but maybe on disk?
        global curr_batch
        curr_range_start = curr_batch * batch_size
        if curr_range_start + batch_size >= num_input_examples:
            return (None, None)
        data_x = np.zeros(shape=(batch_size, num_past_characters, possible_chars))
        data_y = np.zeros(shape=(batch_size, possible_chars))
        for ep in range(batch_size):
            for inc in range(num_past_characters):
                data_x[ep][inc][inverse_char_map[all_text[curr_range_start + ep * step + inc]]] = 1
            data_y[ep][inverse_char_map[all_text[curr_range_start + ep * step + num_past_characters]]] = 1
        curr_batch += 1
        return (data_x, data_y)

    def back_to_text(to_convert):  # gets a one-hot encoded matrix and returns text. We'll need this at the end
        return [chr(char_mapping[numpy.argmax(letter)]) for letter in to_convert]
"""