import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.optimizers import SGD, Adam
# from keras.utils import plot_model

HOME_DIR = os.path.abspath(".") + "/"
TRAIN_DIR = HOME_DIR + 'train_images'
VALIDATION_DIR = HOME_DIR + 'validation_images'
TEST_DIR = HOME_DIR + 'test_images'
TRAIN_LABELS = HOME_DIR + 'train_labels.csv'
VALIDATION_LABELS = HOME_DIR + 'validation_labels.csv'
TEST_LABELS = HOME_DIR + 'test_labels.csv'
EXTENSION = '.png'

# Load training images and labels
os.chdir(TRAIN_DIR)
im_files = os.listdir()
images = [plt.imread(f)[:,:,:3] for f in im_files if f.endswith(EXTENSION)]
print('Number of training images:', len(images))
x_train = np.array(images)
print('x_train shape:', x_train.shape)
y_train_labels = np.genfromtxt(TRAIN_LABELS, delimiter=',')
print('Number of training labels equals number of training images:', len(y_train_labels) == x_train.shape[0])
y_train = keras.utils.to_categorical(y_train_labels, num_classes=2)

# Load validation images and labels
os.chdir(VALIDATION_DIR)
im_files = os.listdir()
images = [plt.imread(f)[:,:,:3] for f in im_files if f.endswith(EXTENSION)]
print('Number of validation images:', len(images))
x_validation = np.array(images)
print('x_validation shape:', x_validation.shape)
y_validation_labels = np.genfromtxt(VALIDATION_LABELS, delimiter=',')
print('Number of validation labels equals number of validation images:', len(y_validation_labels) == x_validation.shape[0])
y_validation = keras.utils.to_categorical(y_validation_labels, num_classes=2)

# Load test images and labels
os.chdir(TEST_DIR)
im_files = os.listdir()
images = [plt.imread(f)[:,:,:3] for f in im_files if f.endswith(EXTENSION)]
print('Number of test images:', len(images))
x_test = np.array(images)
print('x_test shape:', x_test.shape)
y_test_labels = np.genfromtxt(TEST_LABELS, delimiter=',')
print('Number of test labels equals number of test images:', len(y_test_labels) == x_test.shape[0])
y_test = keras.utils.to_categorical(y_test_labels, num_classes=2)

# Create the VGG-like model
model = Sequential()
model.add(Conv2D(32, (3, 3), strides=2, activation='relu', input_shape=(600, 780, 3)))
model.add(Conv2D(32, (3, 3), strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), strides=2, activation='relu'))
model.add(Conv2D(64, (3, 3), strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # two categories
model.summary()
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Learn and evaluate the model
history = model.fit(x_train, y_train, batch_size=32, epochs=12, validation_data=(x_validation, y_validation))
score = model.evaluate(x_test, y_test, batch_size=10)
print('Model Score:', score)

# Write the model, weights and history to files
os.chdir(HOME_DIR)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5") # serialize weights to HDF5
with open('modelhistory.json', 'w') as f:
    json.dump(history.history, f)
plot_model(model, to_file='model.png')
print("Saved model to disk")
