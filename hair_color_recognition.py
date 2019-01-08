import numpy as np
import pickle
import keras
from sklearn.model_selection import train_test_split
import constants
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import data_plotting
from keras.models import load_model


# Reads attribues to train supervised model
labels = pd.read_csv('attribute_list.csv', skiprows=1)
labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
labels.drop(labels.columns[0], axis=1, inplace=True)

#hair_colors = labels["hair_color"]
#labels.hist(column='hair_color')
#plt.show()

with open("saved_variables/all_images.txt", "rb") as fp:  # Unpickling
    images = np.array(pickle.load(fp))

y = np.array([val for i, val in enumerate(labels["hair_color"]) if i + 1 in constants.images])
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.3)

num_classes = 7
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


'''
test = keras.utils.to_categorical(y, 7)
for i in test:
    print(i)

model_accuracy = []

epochs = 5
batch_size = 16

img_width, img_height = 256, 256
#img_width, img_height = None, None

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(16, (7, 7), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (7, 7), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (7, 7), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(7, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)


model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#validation_data=(x_test, y_test)


# Predicts single image and sees weights
#model.save("mnist-model.h5")  # Saves model to file to get inference weights and scores
#model.load_weights("mnist-model.h5")  # Loads weights to see inference - comment out model.fit when doing this
#img_class = model.predict_classes(image)
#prediction = img_class[0]


score = model.evaluate(x_test, y_test, verbose=0)


#preds = model.predict(x_test)
#preds[preds >= 0.5] = 1
#preds[preds < 0.5] = 0
'''

model_accuracy = []
model = load_model('my_model.h5')

score = model.evaluate(x_test, y_test, verbose=0)
model_accuracy.append(score[1])
print('Test loss:', score[0])
print('Test accuracy:', score[1])

