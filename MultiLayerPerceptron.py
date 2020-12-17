import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_MLP(X_train, y_train, X_test, y_test):
    features = len(X_train.columns)
    hidden_layer_nodes = 50
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W1 = tf.Variable(tf.truncated_normal([features, hidden_layer_nodes], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes]))
    z1 = tf.add(tf.matmul(x, W1), b1)
    a1 = tf.nn.relu(z1)
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_nodes, 1], stddev=0.1))
    b2 = tf.Variable(0.)
    z2 = tf.matmul(a1, W2) + b2
    loss = tf.reduce_mean(tf.pow(z2 - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_err = []
    test_err = []
    for i in range(39):
        shuffle_x = X_train.sample(frac=1)
        shuffle_y = y_train.reindex(list(shuffle_x.index.values))
        length = len(shuffle_x)
        for j in range(121, length+2, 120):
            batch_x = shuffle_x.take(range(j-121, min(j,length)))
            batch_y = shuffle_y.take(range(j-121, min(j,length)))
            sess.run(update, feed_dict={x: batch_x, y_: batch_y})
        train_err.append(loss.eval(session=sess, feed_dict={x: X_train, y_: y_train}))
        test_err.append(loss.eval(session=sess, feed_dict={x: X_test, y_: y_test}))
    epochs = [*range(39)]
    return sess.run(W1), sess.run(b1), sess.run(W2), sess.run(b2), epochs, train_err, test_err


def predict_MLP(W1, b1, W2, b2, X_test):
    features = len(X_test.columns)
    x = tf.placeholder(tf.float32, [None, features])
    z1 = tf.add(tf.matmul(x, W1), b1)
    a1 = tf.nn.relu(z1)
    z2 = tf.matmul(a1, W2) + b2
    sess = tf.Session()
    return sess.run(z2, feed_dict={x: X_test})