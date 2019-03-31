# coding: utf8
import tensorflow as tf

x = tf.Variable([1, 2])
y = tf.constant([3, 3])

sub = tf.subtract(x, y)
add = tf.add(x, sub)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # 初始化所有操作
    print sess.run(sub)
