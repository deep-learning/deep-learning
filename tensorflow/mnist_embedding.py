import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

MNIST_PATH = '~/data/mnist'
LOG_DIR = '/tmp/tensorflow/log'
IMAGE_NUM = 10000

mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True)
plot_array = mnist.test.images[:IMAGE_NUM]

# generate meta data
np.savetxt(np.os.path.join(LOG_DIR, 'meta.tsv'), mnist.test.labels[:IMAGE_NUM])

