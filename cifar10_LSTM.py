from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM, Conv2D,ConvLSTM2D, TimeDistributed, MaxPooling2D
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import numpy as np


time_steps = 32
n_rows = 32
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# x_train = [image.reshape((-1,time_steps,n_rows))for image in x_train]


model = Sequential()
# define CNN model
model.add(TimeDistributed(Conv2D(1, (3,3), activation='relu', padding='same', input_shape=(32,32,3))))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
# define LSTM model
model.add(LSTM(32))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train)
# score, acc = model.evaluate(x_test, y_test, batch_size=32)
# print('Test loss = ', score)
# print('Test accuracy = ', acc)
