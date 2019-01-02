import face_recognition
from operator import itemgetter
from PIL import Image, ImageDraw
import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import constants
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn import svm

'''
def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

file_dir = ["dataset/", ".png"]
images = []
edgeness = []
for file_no in constants.images:
    print(file_no)
    file_name = str(file_no).join(file_dir)

    # Load image and extract relevant features
    image = face_recognition.load_image_file(file_name)
    face_locations = face_recognition.face_locations(image)
    face_landmarks_list = face_recognition.face_landmarks(image)[0]  # Returns array so I have to get index 0 to access dictionary

    # Get coordinates for right eye
    right_eye = face_landmarks_list["right_eye"]
    right_eye = ((min(right_eye, key=itemgetter(0))[0] + max(right_eye, key=itemgetter(0))[0]) / 2, max(right_eye, key=itemgetter(0))[1])

    # Get coordinates for left eye
    left_eye = face_landmarks_list["left_eye"]
    left_eye = ((min(left_eye, key=itemgetter(0))[0] + max(left_eye, key=itemgetter(0))[0]) / 2, min(left_eye, key=itemgetter(0))[1])

    distance_between_eyes = calculate_distance(right_eye, left_eye)
    middle_of_eyes = [(left_eye[0]+right_eye[0])/2, (left_eye[1]+right_eye[1])/2]

    top_left_point = (middle_of_eyes[0] - distance_between_eyes, middle_of_eyes[1] + (distance_between_eyes / 7))
    bottom_right_point = (middle_of_eyes[0] + distance_between_eyes, middle_of_eyes[1] + (distance_between_eyes / 7) + (distance_between_eyes / 2))

    area_of_interest = rgb2gray(image[int(top_left_point[1]):int(bottom_right_point[1]), int(top_left_point[0]):int(bottom_right_point[0])])

    current_edgeness = []
    for image_line in range(len(area_of_interest)):
        for pixel_value in range(len(area_of_interest[image_line])):

            try:
                g_i_plus_one = area_of_interest[image_line][pixel_value + 1]
            except IndexError:
                g_i_plus_one = 0
            try:
                g_i_minus_one = area_of_interest[image_line][pixel_value - 1]
            except IndexError:
                g_i_minus_one = 0
            try:
                g_j_plus_one = area_of_interest[image_line + 1][pixel_value]
            except IndexError:
                g_j_plus_one = 0
            try:
                g_j_minus_one = area_of_interest[image_line - 1][pixel_value]
            except IndexError:
                g_j_minus_one = 0

            if image_line == 0:
                g_j_minus_one = 0
            if image_line == len(area_of_interest) - 1:
                g_j_plus_one = 0
            if pixel_value == 0:
                g_i_minus_one = 0
            if pixel_value == len(area_of_interest[image_line] - 1):
                g_i_plus_one = 0

            current_edgeness.append(abs(g_i_plus_one - g_i_minus_one) + abs(g_j_plus_one - g_j_minus_one))

    edgeness.append([np.mean(current_edgeness)])
    del current_edgeness[:]

#For ease of access arrays were saved into separate files
with open("edgeness_values.txt", "wb") as fp:  # Pickling
    pickle.dump(edgeness, fp)
'''

with open("edgeness_values.txt", "rb") as fp:
    edgeness = pickle.load(fp)

# Reads attribues to train supervised model
labels = pd.read_csv('attribute_list.csv', skiprows=1)
labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
labels.drop(labels.columns[0], axis=1, inplace=True)

y = np.array([0 if val == -1 else 1 for i, val in enumerate(labels["eyeglasses"]) if i + 1 in constants.images])

x_train, x_test, y_train, y_test = train_test_split(edgeness, y, test_size=0.3)

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(x_train, y_train)
predicted = logreg.predict(x_test)
print("logistic regression")
print(str(accuracy_score(y_test, predicted)))

clf = svm.SVC(kernel='sigmoid')  # Creates SVM to train
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print("svm")
print(str(accuracy_score(y_test, predicted)))
'''
sorted_colors = [z for z, x in sorted(zip(y,edgeness))]
plt.scatter([i for i in range(len(constants.images))], sorted(edgeness), c=sorted_colors)
plt.show()
'''



