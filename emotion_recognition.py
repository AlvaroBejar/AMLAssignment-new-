import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
import constants
import face_recognition
from sklearn.metrics import accuracy_score

'''
file_dir = ["dataset/", ".png"]
images = []
for file_no in constants.images:
    file_name = str(file_no).join(file_dir)
    image = face_recognition.load_image_file(file_name)
    face_landmarks_list = face_recognition.face_landmarks(image)[0]  # Returns array so I have to get index 0 to access dictionary

    # Get mouth area
    mouth = face_landmarks_list["top_lip"] + face_landmarks_list["bottom_lip"]
    mouth = [list(elem) for elem in mouth]
    images.append(mouth)

#For ease of access arrays were saved into separate files
with open("mouth_images_numpy.txt", "wb") as fp:  # Pickling
    pickle.dump(images, fp)
'''

# Reads attributes to train supervised model
labels = pd.read_csv('attribute_list.csv', skiprows=1)
labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
labels.drop(labels.columns[0], axis=1, inplace=True)

# Load data containing mouth features
with open("saved_variables/mouth_images_numpy.txt", "rb") as fp:  # Unpickling
    images = np.array(pickle.load(fp))

# Puts label data into correct format
y = np.array([0 if val == -1 else 1 for i, val in enumerate(labels["smiling"]) if i + 1 in constants.images])
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.3)

# Reformats data from 3d to 2d for x_train and x_test
nsamples_train, nx_train, ny_train = x_train.shape
x_train_flattened = x_train.reshape((nsamples_train, nx_train * ny_train))

nsamples_test, nx_test, ny_test = x_test.shape
x_test_flattened = x_test.reshape((nsamples_test, nx_test * ny_test))

clf = svm.SVC(kernel='sigmoid')  # Creates SVM to train
clf.fit(x_train_flattened, y_train)
predicted = clf.predict(x_test_flattened)
print(str(accuracy_score(y_test, predicted)))
