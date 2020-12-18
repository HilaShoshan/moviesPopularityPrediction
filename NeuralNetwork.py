import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_NN(X_train, y_train, X_test, y_test, regularization=None, optimizer=None):
    features = len(X_train.columns)
    (hidden1_size, hidden2_size) = (100, 50)
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W1 = tf.Variable(tf.truncated_normal([features, hidden1_size], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
    z1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
    z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)
    W3 = tf.Variable(tf.truncated_normal([hidden2_size, 1], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[1]))
    y = tf.matmul(z2, W3) + b3
    loss = get_loss(regularization, y, y_, W3)
    train_step = get_train_step(optimizer, loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    train_err = []
    test_err = []
    for i in range(100):
        shuffle_x = X_train.sample(frac=1)
        shuffle_y = y_train.reindex(list(shuffle_x.index.values))
        length = len(shuffle_x)
        for j in range(121, length + 2, 120):
            batch_x = shuffle_x.iloc[j - 121:min(j, length), :]
            batch_y = shuffle_y.iloc[j - 121:min(j, length), :]
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        print(i, sess.run(loss, feed_dict={x: X_test, y_: y_test}))
        train_err.append(sess.run(loss, feed_dict={x: X_train, y_: y_train}))
        test_err.append(sess.run(loss, feed_dict={x: X_test, y_: y_test}))
    epochs = [*range(100)]
    return sess.run(W1), sess.run(b1), sess.run(W2), sess.run(b2), sess.run(W3), sess.run(b3), epochs, train_err, test_err


def get_loss(regularization, y, y_, W):
    if regularization is None:
        loss = tf.reduce_mean(tf.pow(y - y_, 2))
    elif regularization == "lasso":
        loss = tf.reduce_mean(tf.pow(y - y_, 2)) + 0.1*tf.reduce_sum(tf.abs(W))
    else:  # regularization == "ridge"
        loss = tf.reduce_mean(tf.pow(y - y_, 2)) + 0.1 * tf.nn.l2_loss(W)
    return loss


def get_train_step(optimizer, loss):
    if optimizer is None:
        train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    else:  # optimizer = "adam"
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_step


def predict_NN(W1, b1, W2, b2, W3, b3, X_test):
    features = len(X_test.columns)
    x = tf.placeholder(tf.float32, [None, features])
    z1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)
    y = tf.matmul(z2, W3) + b3
    sess = tf.Session()
    return sess.run(y, feed_dict={x: X_test})
