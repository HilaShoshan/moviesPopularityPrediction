import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def linearRegression(X_train, y_train):
    features = len(X_train.columns)
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([features, 1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(0, 100000):
        sess.run(update, feed_dict={x: X_train, y_: y_train})
        #if i % 10000 == 0:
         #   print('Iteration:', i, ' W:', sess.run(W), ' b:', sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:X_train, y_:y_train}))
    return sess.run(W), sess.run(b)


def predict_linear(W, b, X_test):
    features = len(X_test.columns)
    x = tf.placeholder(tf.float32, [None, features])
    # y_ = tf.placeholder(tf.float32, [None, 1])
    y_pred = tf.matmul(x, W) + b
    sess = tf.Session()
    return sess.run(y_pred, feed_dict={x: X_test})



