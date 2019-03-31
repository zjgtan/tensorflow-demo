# coding: utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

batch_size = 32
n_batch = mnist.train.num_examples / batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random.normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.random.normal([500], stddev=0.1))
o1 = tf.nn.tanh(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.random.normal([500, 500], stddev=0.1))
b2 = tf.Variable(tf.random.normal([500], stddev=0.1))
o2 = tf.nn.tanh(tf.matmul(o1, W2) + b2)



W3 = tf.Variable(tf.random.normal([500, 10], stddev=0.1))
b3 = tf.Variable(tf.random.normal([10], stddev=0.1))
prediction = tf.nn.softmax(tf.matmul(o2, W3) + b3)

# L2正则
regularizer = tf.contrib.layers.l2_regularizer(0.001)
regularization = regularizer(W1) + regularizer(W2) + regularizer(b1) + regularizer(b2)

#loss = tf.reduce_mean(tf.square(y - prediction))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction)) + \
#        regularization

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = {x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels})
        print epoch, acc

