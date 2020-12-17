import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_linreg(X_train, y_train, X_test, y_test, regularization=None, optimizer=None):
    features = len(X_train.columns)
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([features, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    loss = get_loss(regularization, y, y_, W)
    update = get_train_step(optimizer, loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_err = []
    test_err = []
    for i in range(7):
        shuffle_x = X_train.sample(frac=1)
        shuffle_y = y_train.reindex(list(shuffle_x.index.values))
        length = len(shuffle_x)
        for j in range(121, length+2, 120):
            batch_x = shuffle_x.take(range(j-121, min(j,length)))
            batch_y = shuffle_y.take(range(j-121, min(j,length)))
            sess.run(update, feed_dict={x: batch_x, y_: batch_y})
        print('Iteration:', i, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:', loss.eval(session=sess, feed_dict={x: X_train, y_: y_train}))
        train_err.append(loss.eval(session=sess, feed_dict = {x:X_train, y_:y_train}))
        test_err.append(loss.eval(session=sess, feed_dict = {x:X_test, y_:y_test}))
    epochs = [*range(7)]
    return sess.run(W), sess.run(b), epochs, train_err, test_err


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


def predict_linreg(W, b, X_test):
    features = len(X_test.columns)
    x = tf.placeholder(tf.float32, [None, features])
    y_pred = tf.matmul(x, W) + b
    sess = tf.Session()
    return sess.run(y_pred, feed_dict={x: X_test})



