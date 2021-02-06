import numpy as np
import pandas as pd
import re

from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential , load_model
from keras.layers import Dense , Dropout , LSTM , Bidirectional
from keras.preprocessing.sequence import pad_sequences

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def train_RNN(df):
    
    df['Sentiment'] = [1 if x > 4 else 0 for x in df.popularity]
    X, y = (df['overview'].values.astype("str"), df['Sentiment'].values)

    tk = Tokenizer(lower = True)
    tk.fit_on_texts(X)
    X_seq = tk.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=100, padding='post')
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.25, random_state = 1)

    batch_size = 64
    X_train2 = X_train[batch_size:]
    y_train2 = y_train[batch_size:]
    X_valid = X_train[:batch_size]
    y_valid = y_train[:batch_size]

    # LSTM Model

    vocabulary_size = len(tk.word_counts.keys())+1
    max_words = 100
    embedding_size = 32
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fitting the LSTM model
    model.fit(X_train2, y_train2, validation_data = (X_valid,y_valid) , batch_size=batch_size, epochs=10)

    scores = model.evaluate(X_train,y_train,verbose=0)