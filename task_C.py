import pandas as pd
import face_recognition
from PIL import Image
from operator import itemgetter
import constants
from sklearn.model_selection import train_test_split
import pickle
from sklearn import svm
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
from keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt


# Reads attribues to train supervised model
data = pd.read_csv('attribute_list.csv', skiprows=1)
data.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
data.drop(data.columns[0], axis=1, inplace=True)

'''
# Load image and extract relevant features
image = face_recognition.load_image_file("dataset/2.png")
face_locations = face_recognition.face_locations(image)
face_landmarks_list = face_recognition.face_landmarks(image)[0]  # Returns array so I have to get index 0 to access dictionary

# Get mouth area
mouth = face_landmarks_list["top_lip"] + face_landmarks_list["bottom_lip"]
area_of_interest = image[min(mouth, key=itemgetter(1))[1]:max(mouth, key=itemgetter(1))[1], min(mouth, key=itemgetter(0))[0]:max(mouth, key=itemgetter(0))[0]]

# Get face area
face_image = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
'''

with open("mouth_images_numpy.txt", "rb") as fp:  # Unpickling
    #images = np.stack(pickle.load(fp), axis=0)
    #images = np.array(pickle.load(fp))
    images = np.array(pickle.load(fp))

y = np.array([0 if val == -1 else 1 for i, val in enumerate(data["smiling"]) if i + 1 in constants.images])
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.3)

from sklearn.metrics import accuracy_score

nsamples_train, nx_train, ny_train = x_train.shape
x_train_flattened = x_train.reshape((nsamples_train, nx_train * ny_train))

nsamples_test, nx_test, ny_test = x_test.shape
x_test_flattened = x_test.reshape((nsamples_test, nx_test * ny_test))

clf = svm.SVC(kernel='sigmoid')
clf.fit(x_train_flattened, y_train)
predicted = clf.predict(x_test_flattened)
print(str(accuracy_score(y_test, predicted)))


'''
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
model.add(Dense(1, activation='relu', input_shape=input_shape))
model.add(Dense(10, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

#Comment next 4 lines
# Predicts single image and sees weights
model.save("mnist-model.h5")  # Saves model to file to get inference weights and scores
model.load_weights("mnist-model.h5")  # Loads weights to see inference - comment out model.fit when doing this
img_class = model.predict_classes(image)
prediction = img_class[0]


score = model.evaluate(x_test, y_test, verbose=0)
model_accuracy.append(score[1])
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''