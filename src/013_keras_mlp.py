import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np


X_train = np.random.random((1000, 20))
Y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)))
print(Y_train)