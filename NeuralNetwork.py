import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def NN(df_x, df_y):
    features = len(df_x.columns)
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
    y = tf.nn.softmax(tf.matmul(z2, W3) + b3)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(1000):
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

