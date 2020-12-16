import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_linreg(X_train, y_train):  # on LinearRegression file
    features = len(X_train.columns)
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([features, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(50):
      print('Iteration:', i, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:X_train, y_:y_train}))
      shuffle_x = X_train.sample(frac=1)
      shuffle_y = y_train.reindex(list(shuffle_x.index.values))
      length = len(shuffle_x)
      for j in range(121, length+2, 120):
        batch_x = shuffle_x.take(range(j-121, min(j,length)))
        batch_y = shuffle_y.take(range(j-121, min(j,length)))
        sess.run(update, feed_dict={x: batch_x, y_: batch_y})
    return sess.run(W), sess.run(b)


def predict_linreg(W, b, X_test):
    features = len(X_test.columns)
    x = tf.placeholder(tf.float32, [None, features])
    # y_ = tf.placeholder(tf.float32, [None, 1])
    y_pred = tf.matmul(x, W) + b
    sess = tf.Session()
    return sess.run(y_pred, feed_dict={x: X_test})




