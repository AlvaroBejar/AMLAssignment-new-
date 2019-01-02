import cv2
import face_recognition
from PIL import Image
import matplotlib.colors
import numpy as np
import constants
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

'''
file_dir = ["dataset/", ".png"]
images = []
for file_no in constants.images:
    file_name = str(file_no).join(file_dir)

    image = np.array(face_recognition.load_image_file(file_name))
    blur = np.array(cv2.bilateralFilter(image, 9, 75, 75))

    original = matplotlib.colors.rgb_to_hsv(image)
    blurred = matplotlib.colors.rgb_to_hsv(blur)

    images.append([np.mean(original - blurred)])

#For ease of access arrays were saved into separate files
with open("cartoon_mean.txt", "wb") as fp:  # Pickling
    pickle.dump(images, fp)
'''

# Load data containing mouth features
with open("cartoon_mean.txt", "rb") as fp:  # Unpickling
    images = np.array(pickle.load(fp))

# Reads attributes to train supervised model
labels = pd.read_csv('attribute_list.csv', skiprows=1)
labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
labels.drop(labels.columns[0], axis=1, inplace=True)

y = np.array([0 if val == -1 else 1 for i, val in enumerate(labels["human"]) if i + 1 in constants.images])
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.3)

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(x_train, y_train)
predicted = logreg.predict(x_test)
print(str(accuracy_score(y_test, predicted)))
