import cv2
import face_recognition
from PIL import Image
import matplotlib.colors
import numpy as np
import constants
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import itertools
from sklearn.metrics import confusion_matrix
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


# Saves computed metrics to a file for faster execution
def extract_variables(file_dir):
    #file_dir = ["dataset/", ".png"]
    images = []
    for file_no in range(1, 101):
        file_name = str(file_no).join(file_dir)
        image = np.array(face_recognition.load_image_file(file_name))
        blur = np.array(cv2.bilateralFilter(image, 6, 75, 75))  # Filters image

        # Converts image to HSV
        original = matplotlib.colors.rgb_to_hsv(image)
        blurred = matplotlib.colors.rgb_to_hsv(blur)

        images.append([np.mean(original - blurred)])  # Subtracts images and calculates mean

    return images


# Returns an array with the labels needed for the specific classification task
def get_labels_for_task(column):
    labels = pd.read_csv('attribute_list.csv', skiprows=1)
    labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
    labels.drop(labels.columns[0], axis=1, inplace=True)
    return np.array([0 if val == -1 else 1 for i, val in enumerate(labels[column]) if i + 1 in constants.images])


# Logistic regressor with set parameters
def logistic_regressor(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression(C=1, solver='liblinear', multi_class='ovr')
    logreg.fit(x_train, y_train)
    logreg.predict(x_test)
    #print(str(accuracy_score(y_test, predicted)))
    return logreg



var = extract_variables(["testing_dataset/", ".png"])

images = load_data("saved_variables/cartoon_mean.txt")
y = get_labels_for_task("human")
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.3)
log = logistic_regressor(x_train, y_train, x_test, y_test)

predicted = log.predict_proba(var)
predicted = [[i + 1, val[1]] for i, val in enumerate(predicted)]
print(predicted)

with open("human_detection_C.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(predicted)

'''
class_names = ["Glasses", "No glasses"]
cnf_matrix = confusion_matrix(y_test, predicted)
data_plotting.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.show()
'''
