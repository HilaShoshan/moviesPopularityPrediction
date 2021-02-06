import numpy as np
import pandas as pd
import re

from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Bidirectional

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

stopwords = stopwords.words('english')
newStopWords = ['', ' ', '  ', '   ', '    ', ' s']
stopwords.extend(newStopWords)
stopwords.remove('no')
stopwords.remove('not')
stopwords.remove('very')
stop_words = set(stopwords)


def clean_doc(doc, vocab=None):
    tokens = word_tokenize(doc)
    tokens = [re.sub('[^a-zA-Z]', ' ', word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    if vocab:
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
    return tokens


# adding words to vocabulary
def add_doc_to_vocab(text, vocab):
    tokens = clean_doc(text)
    vocab.update(tokens)


vocab = Counter()
df = pd.read_csv('tmdb_5000_movies.csv')
median = df['popularity'].median() * 3
X = df['overview']
df['Sentiment'] = np.where(df['popularity'] < median, 0, 1)
y = df['Sentiment']
y = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
del df, X, y

len_train = len(X_train)
for i in range(len_train):
    text = str(X_train.iloc[i])
    add_doc_to_vocab(text, vocab)

print(len(vocab))
print(vocab.most_common(20))

min_occurance = 2
tokens = [k for k, c in vocab.items() if (c >= min_occurance & len(k) > 1)]
print(len(tokens))


def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


save_list(tokens, 'vocab.txt')

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

train_doc = []
for i in range(len_train):
    text = str(X_train.iloc[i])
    doc = clean_doc(text, vocab)
    train_doc.append(doc)

test_doc = []
len_test = len(X_test)
for i in range(len_test):
    text = X_test.iloc[i]
    doc = clean_doc(text, vocab)
    test_doc.append(doc)

index_train = []
for i in range(len_train):
    if len(train_doc[i]) == 0:
        index_train.append(i)

index_test = []
for i in range(len_test):
    if len(test_doc[i]) == 0:
        index_test.append(i)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_doc)

X_train = tokenizer.texts_to_matrix(train_doc, mode='binary')
X_test = tokenizer.texts_to_matrix(test_doc, mode='binary')
n_words = X_test.shape[1]

# LSTM Model
model = Sequential()
model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(None, n_words)))
model.add(Dropout(0.2))
model.add(Dense(units=50, input_dim=100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fitting the LSTM model
model.fit(X_train.reshape((-1, 1, n_words)), y_train, epochs=10, batch_size=100)

# finding test loss and test accuracy
loss_rnn, acc_rnn = model.evaluate(X_test.reshape((-1, 1, n_words)), y_test, verbose=0)
print(loss_rnn)
print(acc_rnn)
# df = pd.read_csv('tmdb_5000_movies.csv')
# for i in df['overview']:
#     new_doc = str(i)
#     print(new_doc)
#     new_doc = tokenizer.texts_to_matrix(new_doc)
#     pred = model_rnn.predict(new_doc.reshape((-1, 1, n_words)))
#     print('\n{} stars'.format(pred))