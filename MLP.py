import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import cifar10

# Generate dummy data
import numpy as np
# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

batch_size = 32
num_classes = 10
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# flattening images (32, 32, 3) to 3072 vector
num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)
# normalize inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential()
model.add(Dense(num_pixels, activation='relu', input_dim=3072))
model.add(Dropout(0.5))
model.add(Dense(num_pixels, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=batch_size)
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score = ',score)
print('Test accuracy = ', acc)
