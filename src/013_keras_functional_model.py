from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D


X_train, Y_train, X_test, Y_test = cifar10.load_data()
