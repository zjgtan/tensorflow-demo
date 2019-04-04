# coding: utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

batch_size = 32
n_batch = mnist.train.num_examples / batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

dense1 = tf.layers.dense(
        inputs = x, 
        units = 500, 
        activation = tf.nn.tanh,
        use_bias = True,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))

dense2 = tf.layers.dense(
        inputs = dense1, 
        units = 500, 
        activation = tf.nn.tanh,
        use_bias = True,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))

prediction = tf.layers.dense(
        inputs = dense2, 
        units = 10, 
        activation = tf.nn.softmax,
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.003))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            print type(batch_xs)
            sess.run(train_step, feed_dict = {x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels})
        print epoch, acc

