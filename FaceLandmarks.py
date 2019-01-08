import face_recognition
from operator import itemgetter
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import itertools

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

class FaceLandmarks:

    def __init__(self, file_name):
        self.image = face_recognition.load_image_file(file_name)
        self.face_locations = face_recognition.face_locations(self.image)
        self.face_landmarks_list = face_recognition.face_landmarks(self.image)[0]
        self.right_eye = self.get_right_eye()
        self.left_eye = self.get_left_eye()
        self.nose_tip = self.get_nose_tip()
        self.lip = self.get_lip()
        self.chin = self.get_chin()
        self.middle_of_eyes = self.get_middle_of_eyes()
        self.top_of_head = self.get_top_of_head()
        self.right_face_side = self.get_face_sides()[0]
        self.left_face_side = self.get_face_sides()[1]
        self.distance_between_eyes = calculate_distance(self.right_eye, self.left_eye)

    def get_right_eye(self):
        right_eye = self.face_landmarks_list["right_eye"]
        return ((min(right_eye, key=itemgetter(0))[0] + max(right_eye, key=itemgetter(0))[0]) / 2, max(right_eye, key=itemgetter(0))[1])

    def get_left_eye(self):
        left_eye = self.face_landmarks_list["left_eye"]
        return ((min(left_eye, key=itemgetter(0))[0] + max(left_eye, key=itemgetter(0))[0]) / 2, min(left_eye, key=itemgetter(0))[1])

    def get_nose_tip(self):
        return max(self.face_landmarks_list["nose_tip"], key=itemgetter(1))

    def get_lip(self):
        bottom_lip = self.face_landmarks_list["bottom_lip"]
        return sorted(bottom_lip, key=lambda tup: tup[0])[int(len(bottom_lip) / 2)]

    def get_chin(self):
        return max(self.face_landmarks_list["chin"], key=itemgetter(1))

    def get_middle_of_eyes(self):
        return ((self.left_eye[0] + self.right_eye[0]) / 2, (self.left_eye[1] + self.right_eye[1]) / 2)

    def get_top_of_head(self):
        return (self.middle_of_eyes[0], self.face_locations[0][3])

    def get_face_sides(self):
        # Get coordinates for sides of face based on height of lip
        right_side_ordered = sorted([coords for coords in self.face_landmarks_list["chin"] if coords[0] > self.lip[0]], key=lambda x: abs(self.lip[1] - x[1]))
        left_side_ordered = sorted([coords for coords in self.face_landmarks_list["chin"] if coords[0] < self.lip[0]], key=lambda x: abs(self.lip[1] - x[1]))

        x_lip_line = np.array([list(self.lip), [self.lip[0] + 1, self.lip[1]]])

        # Convert coordinates to numpy array
        right_side_coords = np.array([list(right_side_ordered[0]), list(right_side_ordered[1])])
        left_side_coords = np.array([list(left_side_ordered[0]), list(left_side_ordered[1])])

        t1, s1 = np.linalg.solve(np.array([right_side_coords[1] - right_side_coords[0], x_lip_line[0] - x_lip_line[1]]).T, x_lip_line[0] - right_side_coords[0])
        t2, s2 = np.linalg.solve(np.array([left_side_coords[1] - left_side_coords[0], x_lip_line[0] - x_lip_line[1]]).T, x_lip_line[0] - left_side_coords[0])

        # Calculate intercepts
        right_side = list(map(int, (1 - t1) * right_side_coords[0] + t1 * right_side_coords[1]))
        left_side = list(map(int, (1 - t2) * left_side_coords[0] + t2 * left_side_coords[1]))

        return [right_side, left_side]

    def extract_growth_ratios(self):
        # Calculate ratios
        # Ratio 1: D(left_eye, right_eye)/D(middle_of_eyes, nose)
        r1 = calculate_distance(self.left_eye, self.right_eye) / calculate_distance(self.middle_of_eyes, self.nose_tip)

        # Ratio 2: D(left_eye, right_eye)/D(middle_of_eyes, lip)
        r2 = calculate_distance(self.left_eye, self.right_eye) / calculate_distance(self.middle_of_eyes, self.lip)

        # Ratio 3: D(left_eye, right_eye)/D(middle_of_eyes, chin)
        r3 = calculate_distance(self.left_eye, self.right_eye) / calculate_distance(self.middle_of_eyes, self.chin)

        # Ratio 4: D(middle_of_eyes, nose)/D(middle_of_eyes, lip)
        r4 = calculate_distance(self.middle_of_eyes, self.nose_tip) / calculate_distance(self.middle_of_eyes, self.lip)

        # Ratio 5: D(middle_of_eyes, lip)/D(middle_of_eyes, chin)
        r5 = calculate_distance(self.middle_of_eyes, self.lip) / calculate_distance(self.middle_of_eyes, self.chin)

        # Ratio 6: D(left_eye, right_eye)/D(top_of_head, chin)
        r6 = calculate_distance(self.middle_of_eyes, self.lip) / calculate_distance(self.top_of_head, self.chin)

        # Ratio 7: D(left_side, right_side)/D(top_of_head, chin)
        r7 = calculate_distance(self.right_face_side, self.left_face_side) / calculate_distance(self.top_of_head, self.chin)

        return [r1, r2, r3, r4, r5, r6, r7]

    def get_wrinkle_densities(self):
        # Get wrinkle areas
        # Forehead area wrinkles
        forehead_top_left = (self.middle_of_eyes[0] - (2 / 3) * self.distance_between_eyes, self.middle_of_eyes[1] - (self.distance_between_eyes * 0.75) - (1 / 3) * self.distance_between_eyes)
        forehead_bottom_right = (self.middle_of_eyes[0] + (2 / 3) * self.distance_between_eyes, self.middle_of_eyes[1] - (self.distance_between_eyes * 0.75))
        forehead_area = self.image[int(forehead_top_left[1]):int(forehead_bottom_right[1]), int(forehead_top_left[0]):int(forehead_bottom_right[0])]
        forehead_edges = cv2.Canny(forehead_area, 100, 200)
        forehead_wrinkle_density = np.count_nonzero(forehead_edges) / (forehead_edges.shape[0] * forehead_edges.shape[1])

        # Get wrinkles at right and left of eyes, right eye is actually left based on face, but naming has been made on portrait face
        right_wrinkles_top_left = (self.right_eye[0] + (5 / 12) * self.distance_between_eyes, self.right_eye[1])
        right_wrinkles_bottom_right = (self.right_eye[0] + (5 / 12) * self.distance_between_eyes + (1 / 6) * self.distance_between_eyes, self.right_eye[1] + (1 / 4) * self.distance_between_eyes)
        right_eye_wrinkle_area = self.image[int(right_wrinkles_top_left[1]):int(right_wrinkles_bottom_right[1]), int(right_wrinkles_top_left[0]):int(right_wrinkles_bottom_right[0])]
        right_eye_wrinkle_edges = cv2.Canny(right_eye_wrinkle_area, 100, 200)
        right_eye_wrinkle_density = np.count_nonzero(right_eye_wrinkle_edges) / (right_eye_wrinkle_edges.shape[0] * right_eye_wrinkle_edges.shape[1])

        left_wrinkles_top_left = (self.left_eye[0] - (5 / 12) * self.distance_between_eyes - (1 / 6) * self.distance_between_eyes, self.left_eye[1])
        left_wrinkles_bottom_right = (self.left_eye[0] - (5 / 12) * self.distance_between_eyes, self.left_eye[1] + (1 / 4) * self.distance_between_eyes)
        left_eye_wrinkle_area = self.image[int(left_wrinkles_top_left[1]):int(left_wrinkles_bottom_right[1]), int(left_wrinkles_top_left[0]):int(left_wrinkles_bottom_right[0])]
        left_eye_wrinkle_edges = cv2.Canny(left_eye_wrinkle_area, 100, 200)
        left_eye_wrinkle_density = np.count_nonzero(left_eye_wrinkle_edges) / (left_eye_wrinkle_edges.shape[0] * left_eye_wrinkle_edges.shape[1])
        average_eye_wrinkle_density = np.mean([right_eye_wrinkle_density, left_eye_wrinkle_density])

        # Get cheek wrinkle densities
        right_cheek_wrinkle_density = self.get_cheek_densities(self.right_eye)
        left_cheek_wrinkle_density = self.get_cheek_densities(self.left_eye)
        average_cheek_wrinkle_density = np.mean([right_cheek_wrinkle_density, left_cheek_wrinkle_density])

        return [forehead_wrinkle_density, average_eye_wrinkle_density, average_cheek_wrinkle_density]

    def get_cheek_densities(self, eye):
        cheek_top_left = (eye[0] + (1 / 12) * self.distance_between_eyes, eye[1] + (1 / 8) * self.distance_between_eyes)
        cheek_bottom_right = (eye[0] + (1 / 12) * self.distance_between_eyes, eye[1] + (1 / 8) * self.distance_between_eyes + (1 / 4) * self.distance_between_eyes)
        cheek_wrinkle_area = self.image[int(cheek_top_left[1]):int(cheek_bottom_right[1]), int(cheek_top_left[0]):int(cheek_bottom_right[0])]
        cheek_wrinkle_edges = cv2.Canny(cheek_wrinkle_area, 100, 200)
        return np.count_nonzero(cheek_wrinkle_edges) / (cheek_wrinkle_edges.shape[0] * cheek_wrinkle_edges.shape[1])
