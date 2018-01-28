from tensorflow.contrib.keras.api import keras
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


# input = K.placeholder(shape=(10, 32))
# input2 = tf.placeholder(tf.float32, shape=(10, 32))
# print(input)
# print(input2)

# model = Sequential()
# model.add(Dense(units=16, input_dim=784))
# model.add(Activation('softmax'))

model = Sequential([
    Dense(units=64, input_shape=(784,), activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              # optimizer='sgd',
              optimizer=keras.optimizers.SGD(lr=0.02, momentum=0.8, nesterov=True),
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           verbose=0,
                           mode='auto')

model.fit()
