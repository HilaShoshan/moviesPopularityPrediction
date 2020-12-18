import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_NN(X_train, y_train, X_test, y_test, num_epochs=50, num_hidden_layers=2, hidden_sizes=(100,50), regularization=None, optimizer=None):
    """
    implements a multi-layers neural network with the given parameters.
    :param hidden_sizes: a tuple of size num_hidden_layer param, that represents the number of neurons of each layer.
    :return: the weights and biases after training,
                and lists of epochs, train error and test error for plotting a graph on main
    """
    if len(hidden_sizes) != num_hidden_layers:
        print("size of tuple does not fit the num_hidden_layer")
        return

    features = len(X_train.columns)
    Ws = get_Ws(num_hidden_layers, hidden_sizes, features)
    biases = get_biases(hidden_sizes)
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    activations = get_activations(x, Ws, biases)
    y = activations[-1]

    loss = get_loss(regularization, y, y_, Ws[-1])
    train_step = get_train_step(optimizer, loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train_err = []
    test_err = []
    epochs = [*range(num_epochs)]

    for i in range(num_epochs):
        shuffle_x = X_train.sample(frac=1)
        shuffle_y = y_train.reindex(list(shuffle_x.index.values))
        length = len(shuffle_x)
        for j in range(121, length + 2, 120):
            batch_x = shuffle_x.iloc[j - 121:min(j, length), :]
            batch_y = shuffle_y.iloc[j - 121:min(j, length), :]
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        print(i, "train loss: ", sess.run(loss, feed_dict={x: X_train, y_: y_train}), "test loss: ", sess.run(loss, feed_dict={x: X_test, y_: y_test}))
        train_err.append(sess.run(loss, feed_dict={x: X_train, y_: y_train}))
        test_err.append(sess.run(loss, feed_dict={x: X_test, y_: y_test}))

    ret_W = []
    ret_b = []
    for i in range(len(Ws)):
        ret_W.append(sess.run(Ws[i]))
        ret_b.append(sess.run(biases[i]))
    return ret_W, ret_b, epochs, train_err, test_err


def get_Ws(num_hidden_layers, hidden_sizes, features):
    ans = []  # list of all Ws to return
    for i in range(num_hidden_layers):
        if i == 0:  # this is the first weights matrix
            W = tf.Variable(tf.truncated_normal([features, hidden_sizes[i]], stddev=0.1))
        else:
            W = tf.Variable(tf.truncated_normal([hidden_sizes[i-1], hidden_sizes[i]], stddev=0.1))
        ans.append(W)
    ans.append(tf.Variable(tf.truncated_normal([hidden_sizes[-1], 1], stddev=0.1)))  # the last W (to the output layer)
    return ans


def get_biases(hidden_sizes):
    ans = []  # list of all the biases to return
    for size in hidden_sizes:
        b = tf.Variable(tf.constant(0.1, shape=[size]))
        ans.append(b)
    ans.append(tf.Variable(tf.constant(0.1, shape=[1])))  # the last bias (for the output layer)
    return ans


def get_activations(x, Ws, biases):
    """
        get all layers outputs: use relu activation function in hidden, and no activation on the last layer
    """
    ans = []
    for i in range(len(Ws)-1):  # Ws size = biases size always
        if i == 0:
            z = tf.add(tf.matmul(x, Ws[i]), biases[i])
        else:
            z = tf.add(tf.matmul(ans[i-1], Ws[i]), biases[i])
        ans.append(tf.nn.relu(z))
    ans.append(tf.matmul(ans[-1], Ws[-1]) + biases[-1])  # no activation on output layer
    return ans


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


def predict_NN(Ws, biases, X_test):
    features = len(X_test.columns)
    x = tf.placeholder(tf.float32, [None, features])
    outputs = get_activations(x, Ws, biases)
    y = outputs[-1]
    sess = tf.Session()
    return sess.run(y, feed_dict={x: X_test})
