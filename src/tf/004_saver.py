import tensorflow as tf

# The tf.train.Saver class provides methods for saving and restoring models.
# The tf.train.Saver constructor adds save and restore ops to the graph for all, or a specified list,
# of the variables in the graph.
# The Saver object provides methods to run these ops, specifying paths for the checkpoint files to write to or read from.

# TensorFlow saves variables in binary checkpoint files that, roughly speaking, map variable names to tensor values.

v1 = tf.get_variable('v1', shape=[3], initializer=tf.zeros_initializer)
print(v1)
v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)
print(v2)

inc_v1 = v1.assign(v1 + 1)
dec_v2 = v2.assign(v2 - 1)

init_op = tf.global_variables_initializer()

# add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    inc_v1.op.run()
    dec_v2.op.run()

    save_path = saver.save(sess, '/tmp/model.ckpt')
    print('model saved in path: %s' % save_path)

# restoring
tf.reset_default_graph()
v1 = tf.get_variable('v1', shape=[3])
print(v1)
v2 = tf.get_variable('v2', shape=[5], initializer=tf.zeros_initializer)
print(v2)

# You can create as many Saver objects as you want if you need to save and restore different subsets of the model variables.
# add op to save or restore all the variables
saver = tf.train.Saver()
saver_v2 = tf.train.Saver({'v2': v2})
with tf.Session() as sess:
    print('model with v2 restored')
    saver_v2.restore(sess, '/tmp/model.ckpt')
    # v1 is uninitialized
    v1.initializer.run()
    print('v1: %s' % v1.eval())
    print('v2: %s' % v2.eval())

    saver.restore(sess, '/tmp/model.ckpt')
    print('model restored')
    print('v1: %s' % v1.eval())
    print('v2: %s' % v2.eval())



from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file('/tmp/model.ckpt', tensor_name='', all_tensors=True)
chkp.print_tensors_in_checkpoint_file('/tmp/model.ckpt', tensor_name='v1', all_tensors=False)

# todo with name or variable scope??