from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import cv2
import logging
logging.getLogger('tensorflow').disabled = True

(x_train, y_train), (x_test, y_test) = mnist.load_data()

noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape) / 2.5
x_train = (1 - x_train.astype('float32') / 255) - noise
x_train = np.clip(x_train, 0.0, 1.0)
y_train = to_categorical(y_train)

noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape) / 2.5
x_test = (1 - x_test.astype('float32') / 255) - noise
x_test = np.clip(x_test, 0.0, 1.0)
y_test = to_categorical(y_test)

plt.imshow(x_train[10], cmap='gray')
plt.show()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))


# model
filters = [64, 128, 256]
model = Sequential()

model.add(Conv2D(filters[0], (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters[1], (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters[2], (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(GlobalMaxPooling2D())
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.0001, verbose=1, cooldown=1)
tensorboard = TensorBoard(log_dir='my_log_dir', histogram_freq=1)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=25, batch_size=32,
          validation_data=(x_test, y_test), callbacks=[reduce_lr, tensorboard])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


