# Recurrent Neural Network - what we use for sequences
# use overview column as features

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_RNN(df):
    x, y = arrange(df)
    vocab, max_sentence_len = get_vocab(x)  # get all unique words in the whole overviews
    num_options = len(vocab)
    num_features = max_sentence_len
    batch_size = 15
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    num_input_examples = x_train.shape[0]  # number of examples = number of rows in x_train
    curr_batch = 0
    cellsize = 30
    x = tf.placeholder(tf.float32, [None, num_features,  num_options])
    y = tf.placeholder(tf.float32, [None, 1])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cellsize, forget_bias=0.0)
    output, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = output[-1]
    W = tf.Variable(tf.truncated_normal([cellsize, num_options], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_options]))
    z = tf.matmul(last, W) + b
    res = tf.nn.softmax(z)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(res), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(res, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num_of_epochs = 100
    for ephoch in range(num_of_epochs):
        acc = 0
        curr_batch = 0
        while True:
            batch_xs, batch_ys = next_batch(x_train, y_train, curr_batch, num_input_examples, batch_size, num_options, num_features, vocab)
            if batch_xs is None:
                break
            else:
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
                acc += accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
        print("step %d, training accuracy %g" % (ephoch, acc / curr_batch))


def next_batch(x_train, y_train, curr_batch, num_input_examples, batch_size, num_options, num_features, vocab):
    curr_range_start = curr_batch*batch_size
    if curr_range_start + batch_size >= num_input_examples:
        return (None, None)
    data_x = np.zeros(shape=(batch_size, num_features, num_options))
    data_y = np.zeros(shape=(batch_size, 1))
    for ep in range(batch_size):
        example_index = curr_range_start + ep
        overview = x_train.iloc[example_index]['overview']
        words = np.array(overview.split())
        for inc in range(len(words)):  # if it smaller than num_features, so there will remain zeros on the matrix
            word = words[inc]
            voc_index = vocab.index(word)
            data_x[ep][inc][voc_index] = 1
        data_y[ep][0] = y_train.iloc[example_index]['popularity']
    curr_batch += 1
    return (data_x, data_y)


def arrange(df):
    df.dropna(0, subset=['overview'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    x = df[['overview']]
    y = df[['popularity']]
    return x, y


def get_features(x_train, row):
    text = x_train.at[row, 'overview']
    words = np.array(text.split())  # create an array of all the words in the overview
    unique_words = np.unique(words)  # an array of unique words, to use as features
    num_options = len(unique_words)
    encoding_mat = pd.get_dummies(unique_words)
    features = []
    for word in words:
        encode = list(encoding_mat.loc[:, word])
        features.append(encode)  # add the one-hot-encoding vector that represents the word as feature
    return features, num_options


def get_vocab(x):
    vocab = []
    max_sentence_len = 0
    for row in range(x.shape[0]):
        text = x.at[row, 'overview']
        words = np.array(text.split())  # create an array of all the words in the overview
        if len(words) > max_sentence_len:
            max_sentence_len = len(words)
        unique_words = np.unique(words)  # an array of unique words
        for word in unique_words:
            if word not in vocab:
                vocab.append(word)
    return vocab, max_sentence_len