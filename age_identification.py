import face_recognition
import pickle
import constants
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import PIL
from operator import itemgetter
from PIL import Image, ImageDraw
import math
import cv2


# Reads attributes to train supervised model
labels = pd.read_csv('attribute_list.csv', skiprows=1)
labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
labels.drop(labels.columns[0], axis=1, inplace=True)

'''
def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

file_dir = ["dataset/", ".png"]
ratios = []
for file_no in constants.images:
    print(file_no)
    file_name = str(file_no).join(file_dir)
    # Load image, detect face and landmarks
    image = face_recognition.load_image_file(file_name)
    face_locations = face_recognition.face_locations(image)
    #face_image = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1], :]  # Crop image

    try:
        face_landmarks_list = face_recognition.face_landmarks(image)[0]
    except Exception:
        ratios.append([0, 0, 0, 0, 0, 0, 0, labels["hair_color"][file_no - 1]])
        continue

    # Get coordinates for right eye
    right_eye = face_landmarks_list["right_eye"]
    right_eye = ((min(right_eye, key=itemgetter(0))[0] + max(right_eye, key=itemgetter(0))[0]) / 2, max(right_eye, key=itemgetter(0))[1])

    # Get coordinates for left eye
    left_eye = face_landmarks_list["left_eye"]
    left_eye = ((min(left_eye, key=itemgetter(0))[0] + max(left_eye, key=itemgetter(0))[0]) / 2, min(left_eye, key=itemgetter(0))[1])

    nose_tip = max(face_landmarks_list["nose_tip"], key=itemgetter(1))

    chin = max(face_landmarks_list["chin"], key=itemgetter(1))

    # Get lip coordinates
    sorted_lip = sorted(face_landmarks_list["bottom_lip"], key=lambda tup: tup[0])
    lip = sorted_lip[int(len(sorted_lip)/2)]

    # Get coordinates for sides of face based on height of lip
    right_side_ordered = sorted([coords for coords in face_landmarks_list["chin"] if coords[0] > lip[0]], key=lambda x: abs(lip[1] - x[1]))
    left_side_ordered = sorted([coords for coords in face_landmarks_list["chin"] if coords[0] < lip[0]], key=lambda x: abs(lip[1] - x[1]))

    x_lip_line = np.array([list(lip), [lip[0]+1, lip[1]]])

    # Convert coordinates to numpy array
    right_side_coords = np.array([list(right_side_ordered[0]), list(right_side_ordered[1])])
    left_side_coords = np.array([list(left_side_ordered[0]), list(left_side_ordered[1])])

    t1, s1 = np.linalg.solve(np.array([right_side_coords[1]-right_side_coords[0], x_lip_line[0]-x_lip_line[1]]).T, x_lip_line[0]-right_side_coords[0])
    t2, s2 = np.linalg.solve(np.array([left_side_coords[1]-left_side_coords[0], x_lip_line[0]-x_lip_line[1]]).T, x_lip_line[0]-left_side_coords[0])

    # Calculate intercepts
    right_side = list(map(int, (1-t1)*right_side_coords[0] + t1*right_side_coords[1]))
    left_side = list(map(int, (1-t2)*left_side_coords[0] + t2*left_side_coords[1]))

    # Calculate top of head coordinates
    middle_of_eyes = [(left_eye[0]+right_eye[0])/2, (left_eye[1]+right_eye[1])/2]
    top_of_head = [middle_of_eyes[0], face_locations[0][3]]

    # Calculate ratios
    # Ratio 1: D(left_eye, right_eye)/D(middle_of_eyes, nose)
    r1 = calculate_distance(left_eye, right_eye)/calculate_distance(middle_of_eyes, nose_tip)

    # Ratio 2: D(left_eye, right_eye)/D(middle_of_eyes, lip)
    r2 = calculate_distance(left_eye, right_eye)/calculate_distance(middle_of_eyes, lip)

    # Ratio 3: D(left_eye, right_eye)/D(middle_of_eyes, chin)
    r3 = calculate_distance(left_eye, right_eye)/calculate_distance(middle_of_eyes, chin)

    # Ratio 4: D(middle_of_eyes, nose)/D(middle_of_eyes, lip)
    r4 = calculate_distance(middle_of_eyes, nose_tip)/calculate_distance(middle_of_eyes, lip)

    # Ratio 5: D(middle_of_eyes, lip)/D(middle_of_eyes, chin)
    r5 = calculate_distance(middle_of_eyes, lip)/calculate_distance(middle_of_eyes, chin)

    # Ratio 6: D(left_eye, right_eye)/D(top_of_head, chin)
    r6 = calculate_distance(middle_of_eyes, lip)/calculate_distance(top_of_head, chin)

    # Ratio 7: D(left_side, right_side)/D(top_of_head, chin)
    r7 = calculate_distance(right_side, left_side)/calculate_distance(top_of_head, chin)

    # Get wrinkle areas
    distance_between_eyes = calculate_distance(right_eye, left_eye)

    # Forehead area wrinkles
    forehead_top_left = (middle_of_eyes[0] - (2 / 3) * distance_between_eyes, middle_of_eyes[1] - (distance_between_eyes * 0.75) - (1 / 3) * distance_between_eyes)
    forehead_bottom_right = (middle_of_eyes[0] + (2 / 3) * distance_between_eyes, middle_of_eyes[1] - (distance_between_eyes * 0.75))
    forehead_area = image[int(forehead_top_left[1]):int(forehead_bottom_right[1]), int(forehead_top_left[0]):int(forehead_bottom_right[0])]
    forehead_edges = cv2.Canny(forehead_area, 100, 200)
    forehead_wrinkle_density = np.count_nonzero(forehead_edges) / (forehead_edges.shape[0] * forehead_edges.shape[1])

    # Get wrinkles at right and left of eyes, right eye is actually left based on face, but naming has been made on portrait face
    right_wrinkles_top_left = (right_eye[0] + (5/12) * distance_between_eyes, right_eye[1])
    right_wrinkles_bottom_right = (right_eye[0] + (5/12) * distance_between_eyes + (1/6) * distance_between_eyes, right_eye[1] + (1/4) * distance_between_eyes)
    right_eye_wrinkle_area = image[int(right_wrinkles_top_left[1]):int(right_wrinkles_bottom_right[1]), int(right_wrinkles_top_left[0]):int(right_wrinkles_bottom_right[0])]
    right_eye_wrinkle_edges = cv2.Canny(right_eye_wrinkle_area, 100, 200)
    right_eye_wrinkle_density = np.count_nonzero(right_eye_wrinkle_edges) / (right_eye_wrinkle_edges.shape[0] * right_eye_wrinkle_edges.shape[1])

    left_wrinkles_top_left = (left_eye[0] - (5/12) * distance_between_eyes - (1/6) * distance_between_eyes, left_eye[1])
    left_wrinkles_bottom_right = (left_eye[0] - (5/12) * distance_between_eyes, left_eye[1] + (1/4) * distance_between_eyes)
    left_eye_wrinkle_area = image[int(left_wrinkles_top_left[1]):int(left_wrinkles_bottom_right[1]), int(left_wrinkles_top_left[0]):int(left_wrinkles_bottom_right[0])]
    left_eye_wrinkle_edges = cv2.Canny(left_eye_wrinkle_area, 100, 200)
    left_eye_wrinkle_density = np.count_nonzero(left_eye_wrinkle_edges) / (left_eye_wrinkle_edges.shape[0] * left_eye_wrinkle_edges.shape[1])
    average_eye_wrinkle_density = np.mean([right_eye_wrinkle_density, left_eye_wrinkle_density])

    # Get cheekbone areas for wrinkles, naming is inverted as with eyes
    right_cheek_top_left = (right_eye[0] - (1/12) * distance_between_eyes, right_eye[1] + (1/8) * distance_between_eyes)
    right_cheek_bottom_right = (right_eye[0] + (1/12) * distance_between_eyes, right_eye[1] + (1/8) * distance_between_eyes + (1/4) * distance_between_eyes)
    right_cheek_wrinkle_area = image[int(right_cheek_top_left[1]):int(right_cheek_bottom_right[1]), int(right_cheek_top_left[0]):int(right_cheek_bottom_right[0])]
    right_cheek_wrinkle_edges = cv2.Canny(right_cheek_wrinkle_area, 100, 200)
    right_cheek_wrinkle_density = np.count_nonzero(right_cheek_wrinkle_edges) / (right_cheek_wrinkle_edges.shape[0] * right_cheek_wrinkle_edges.shape[1])

    left_cheek_top_left = (left_eye[0] - (1/12) * distance_between_eyes, left_eye[1] + (1/8) * distance_between_eyes)
    left_cheek_bottom_right = (left_eye[0] + (1/12) * distance_between_eyes, left_eye[1] + (1/8) * distance_between_eyes + (1/4) * distance_between_eyes)
    left_cheek_wrinkle_area = image[int(left_cheek_top_left[1]):int(left_cheek_bottom_right[1]), int(left_cheek_top_left[0]):int(left_cheek_bottom_right[0])]
    left_cheek_wrinkle_edges = cv2.Canny(left_eye_wrinkle_area, 100, 200)
    left_cheek_wrinkle_density = np.count_nonzero(left_cheek_wrinkle_edges) / (left_cheek_wrinkle_edges.shape[0] * left_cheek_wrinkle_edges.shape[1])
    average_cheek_wrinkle_density = np.mean([right_cheek_wrinkle_density, left_cheek_wrinkle_density])

    #labels["hair_color"][file_no - 1], append hair color to array

    ratios.append([r1, r2, r3, r4, r5, r6, r7, forehead_wrinkle_density, average_eye_wrinkle_density, average_cheek_wrinkle_density])


with open("age_recognition_with_wrinkles.txt", "wb") as fp:  # Pickling
    pickle.dump(ratios, fp)
'''

# !!!!!! IMPLEMENT WRINKLE ANALYSIS USING EDGE DETECTION !!!!!!
# COMPARE ONLY HAIR WITH FACE DISTANCES WITH WRINKLE ANALYSIS WITH MIX

with open("age_recognition_with_wrinkles.txt", "rb") as fp:
    images = np.array(pickle.load(fp))

# Reads attributes to train supervised model
labels = pd.read_csv('saved_variables/attribute_list.csv', skiprows=1)
labels.columns = ["id", "hair_color", "eyeglasses", "smiling", "young", "human"]
labels.drop(labels.columns[0], axis=1, inplace=True)
y = np.array([0 if val == -1 else 1 for i, val in enumerate(labels["young"]) if i + 1 in constants.images])
#x = np.array([[val] for i, val in enumerate(labels["hair_color"]) if i + 1 in constants.images])
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.3)

from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(images)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_[7])

clf = svm.SVC(kernel='sigmoid')  # Creates SVM to train
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
print(str(accuracy_score(y_test, predicted)))
