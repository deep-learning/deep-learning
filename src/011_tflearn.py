import tflearn
import tflearn.datasets.mnist as mnist
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

X_train, Y_train, X_test, Y_test = mnist.load_data(one_hot=True)
X_train = X_train.reshape([-1, 28, 28, 1])
X_test = X_test.reshape([-1, 28, 28, 1])

# building network
CNN = input_data(shape=[None, 28, 28, 1], name='input')
CNN = conv_2d(CNN, nb_filter=32, filter_size=5, activation='relu', regularizer='L2')
CNN = max_pool_2d(CNN, kernel_size=2)
CNN = local_response_normalization(CNN)
CNN = conv_2d(CNN, 64, 5, activation='relu', regularizer='L2')
CNN = max_pool_2d(CNN, kernel_size=2)
CNN = local_response_normalization(CNN)
CNN = fully_connected(CNN, n_units=1024, activation=None)
CNN = dropout(CNN, keep_prob=0.5)
CNN = fully_connected(CNN, 10, activation='softmax')
CNN = regression(CNN, optimizer='adam', learning_rate='0.0001',
                 loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network=CNN,
                    tensorboard_verbose=0,
                    tensorboard_dir='MNITS_tflearn_logs',
                    checkpoint_path='MNIST_tflearn_checkpoints/checkpoint')
model.fit(X_inputs={'input': X_train},
          Y_targets={'target': Y_train},
          batch_size=64,
          n_epoch=3,
          validation_set=({'input': X_test}, {'target': Y_test}),
          validation_batch_size=128,
          snapshot_step=1000,
          show_metric=True,
          run_id='convnet_mnist')


model.evaluate(X_test, Y_test, batch_size=128)