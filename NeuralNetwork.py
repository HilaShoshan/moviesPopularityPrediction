import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_NN(X_train, y_train, X_test, y_test):
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
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    error = tf.cast(loss, tf.float32)
    train_err = []
    test_err = []
    for i in range(50):
        shuffle_x = X_train.sample(frac=1)
        shuffle_y = y_train.reindex(list(shuffle_x.index.values))
        length = len(shuffle_x)
        for j in range(121, length + 2, 120):
            batch_x = shuffle_x.take(range(j - 121, min(j, length)))
            batch_y = shuffle_y.take(range(j - 121, min(j, length)))
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        print(i, sess.run(error, feed_dict={x: X_test, y_: y_test}))
        train_err.append(sess.run(error, feed_dict={x: X_train, y_: y_train}))
        test_err.append(sess.run(error, feed_dict={x: X_test, y_: y_test}))
    epochs = [*range(50)]
    return sess.run(W1), sess.run(b1), sess.run(W2), sess.run(b2), sess.run(W3), sess.run(b3), epochs, train_err, test_err


def predict_NN(W1, b1, W2, b2, W3, b3, X_test):
    features = len(X_test.columns)
    x = tf.placeholder(tf.float32, [None, features])
    z1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)
    y = tf.matmul(z2, W3) + b3
    sess = tf.Session()
    return sess.run(y, feed_dict={x: X_test})
