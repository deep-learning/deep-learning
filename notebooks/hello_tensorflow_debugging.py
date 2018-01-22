import tensorflow as tf
from tensorflow.python import debug as tf_debug
a = tf.constant([1.0, 4.0], shape=[2,1])
b = tf.constant([2.0, 3.0], shape=[1,2])
c = tf.add(tf.matmul(a,b), tf.constant([5.0, 6.0]))
d = tf.Print(c, [c, 2.0], message="Value of C is:")
with tf.Session() as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(d)