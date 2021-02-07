import numpy as np
import pandas as pd
import re

from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt


def train_RNN(df_x, df_y):
    df_x['overview'] = df_x['overview'].values.astype("str")

    tk = Tokenizer(lower=True)
    tk.fit_on_texts(df_x['overview'].values)
    df_x['overview'] = tk.texts_to_sequences(df_x['overview'].values)
    X_pad = pad_sequences(df_x['overview'].values, maxlen=100, padding='post')
    df_x = pd.concat([df_x, pd.DataFrame(X_pad)], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

    # LSTM Model
    vocabulary_size = len(tk.word_counts.keys()) + 1
    max_words = 100
    embedding_size = 32
    model = Sequential()
    model.add(layers.Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # model.summary()
    # fitting the LSTM model
    history = model.fit(X_train.iloc[:, -100:], y_train.values,
                        validation_data=(X_val.iloc[:, -100:], y_val.values), batch_size=60, epochs=50, shuffle=True)

    X_train['overview'] = model.predict(X_train.iloc[:, -100:])
    print(model.predict(X_train.iloc[:, -100:]))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    scores = model.evaluate(X_test.iloc[:, -100:].values, y_test.values, verbose=2)