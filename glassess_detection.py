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
import csv
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import itertools
import data_plotting

def save_data(file_name, images):
    # For ease of access arrays were saved into separate files
    with open(file_name, "wb") as fp:  # Pickling
        pickle.dump(images, fp)


# Loads data from file with saved variables
def load_data(file_name):
    # Load data containing mouth features
    with open(file_name, "rb") as fp:  # Unpickling
        return np.array(pickle.load(fp))


# Returns an array with the labels needed for the specific classification task
def get_labels_for_task(column):
    labels = pd.read_csv('attribute_list.csv', skiprows=1)
    labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
    labels.drop(labels.columns[0], axis=1, inplace=True)
    return np.array([0 if val == -1 else 1 for i, val in enumerate(labels[column]) if i + 1 in constants.images])


def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def extract_features(file_dir):
    edgeness = []
    for file_no in range(1, 101):
        file_name = str(file_no).join(file_dir)
        # Load image and extract relevant features
        image = face_recognition.load_image_file(file_name)
        try:
            face_landmarks_list = face_recognition.face_landmarks(image)[0]  # Returns array so I have to get index 0 to access dictionary
        except Exception:
            edgeness.append([0])
            continue

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
    return edgeness


# Logistic regressor with set parameters
def logistic_regressor(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression(C=1, solver='liblinear', multi_class='ovr')
    logreg.fit(x_train, y_train)
    logreg.predict(x_test)
    #print(str(accuracy_score(y_test, predicted)))
    return logreg

edgeness = load_data("saved_variables/edgeness_values.txt")
y = get_labels_for_task("eyeglasses")
x_train, x_test, y_train, y_test = train_test_split(edgeness, y, test_size=0.3)

var = extract_features(["testing_dataset/", ".png"])
log = logistic_regressor(x_train, y_train, x_test, y_test)
predicted = log.predict_proba(var)
predicted = [[i + 1, val[1]] for i, val in enumerate(predicted)]


with open("glasses_detection_C.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(predicted)


'''
sorted_colors = [z for z, x in sorted(zip(y,edgeness))]
plt.scatter([i for i in range(len(constants.images))], sorted(edgeness), c=sorted_colors)
plt.show()
'''



