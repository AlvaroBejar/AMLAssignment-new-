from operator import itemgetter
import face_recognition
import constants as const
import math
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.colors


def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

class FeatureExtractor:

    def __init__(self, images):
        self.images = images
        self.extract_features()
        self.extract_face_locations()
        self.dct = {
            "left_eye": self.get_left_eye,
            "right_eye": self.get_right_eye,
            "nose_tip": self.get_nose_tip,
            "lip": self.get_lip,
            "chin": self.get_chin,
            "middle_of_eyes": self.get_middle_of_eyes,
            "top_of_head": self.get_top_of_head,
            "mouth": self.get_mouth,
            "distance_between_eyes" : self.get_distance_between_eyes,
            "edgeness": self.get_edgeness,
            "human_values": self.get_human_features,
            "age_features": self.get_age_features
        }

    def extract_features(self):
        features = []
        for image in self.images:
            features.append(face_recognition.face_landmarks(image)[0])

            '''
            try:
                features.append(face_recognition.face_landmarks(images[i])[0])
            except Exception:
                pass
            '''

        self.face_landmarks_list = features

    def extract_face_locations(self):
        locations = []
        for i, val in enumerate(self.images):
            locations.append(face_recognition.face_locations(val)[0])
            #locations.append(face_recognition.face_locations(images[i - 1])[0])
        self.face_locations = locations

    def get_multiple_features(self, feature_keys):
        return [self.dct[key]() for key in feature_keys]

    def get_left_eye(self):
        left_eyes = []
        for landmarks in self.face_landmarks_list:
            left_eye = landmarks["left_eye"]
            left_eyes.append(((min(left_eye, key=itemgetter(0))[0] + max(left_eye, key=itemgetter(0))[0]) / 2, min(left_eye, key=itemgetter(0))[1]))
        return left_eyes

    def get_right_eye(self):
        right_eyes = []
        for landmarks in self.face_landmarks_list:
            right_eye = landmarks["right_eye"]
            right_eyes.append(((min(right_eye, key=itemgetter(0))[0] + max(right_eye, key=itemgetter(0))[0]) / 2, max(right_eye, key=itemgetter(0))[1]))
        return right_eyes

    def get_nose_tip(self):
        nose_tips = []
        for landmarks in self.face_landmarks_list:
            nose_tips.append(max(landmarks["nose_tip"], key=itemgetter(1)))
        return nose_tips

    def get_lip(self):
        bottom_lips = []
        for landmarks in self.face_landmarks_list:
            bottom_lip = landmarks["bottom_lip"]
            bottom_lips.append(sorted(bottom_lip, key=lambda tup: tup[0])[int(len(bottom_lip) / 2)])
        return bottom_lips

    def get_chin(self):
        get_chins = []
        for landmarks in self.face_landmarks_list:
            get_chins.append(max(landmarks["chin"], key=itemgetter(1)))
        return get_chins

    def get_middle_of_eyes(self):
        right_eyes = self.get_right_eye()
        left_eyes = self.get_left_eye()

        return [((left_eyes[i][0] + right_eyes[i][0]) / 2, (left_eyes[i][1] + right_eyes[i][1]) / 2)
                for i in range(0, len(right_eyes))]

    def get_top_of_head(self):
        middle_of_eyes = self.get_middle_of_eyes()
        coords = []
        for i, val in enumerate(middle_of_eyes):
            coords.append((val[0], self.face_locations[i][3]))
        return coords

    def get_mouth(self):
        mouths = []
        for landmarks in self.face_landmarks_list:
            mouth = landmarks["top_lip"] + landmarks["bottom_lip"]
            mouth = [list(elem) for elem in mouth]
            mouths.append(mouth)
        return mouths

    def get_distance_between_eyes(self):
        distance = []
        right_eyes = self.get_right_eye()
        left_eyes = self.get_left_eye()
        for i in range(len(right_eyes)):
            distance.append(calculate_distance(right_eyes[i], left_eyes[i]))
        return distance

    # Used for glasses detection
    def get_edgeness(self):
        edgeness = []
        middle_of_eyes = self.get_middle_of_eyes()
        distance_between_eyes = self.get_distance_between_eyes()
        for i in range(len(self.images)):
            top_left_point = (middle_of_eyes[i][0] - distance_between_eyes[i], middle_of_eyes[i][1] + (distance_between_eyes[i] / 7))
            bottom_right_point = (middle_of_eyes[i][0] + distance_between_eyes[i], middle_of_eyes[i][1] + (distance_between_eyes[i] / 7) + (distance_between_eyes[i] / 2))

            area_of_interest = rgb2gray(self.images[i][int(top_left_point[1]):int(bottom_right_point[1]), int(top_left_point[0]):int(bottom_right_point[0])])

            current_edgeness = []
            for image_line in range(len(area_of_interest)):
                for pixel_value in range(len(area_of_interest[image_line])):
                    g_j_plus_one = area_of_interest[image_line + 1][pixel_value] if image_line != (len(area_of_interest) - 1) else 0
                    g_j_minus_one = area_of_interest[image_line - 1][pixel_value] if image_line != 0 else 0
                    g_i_plus_one = area_of_interest[image_line][pixel_value + 1] if pixel_value != (len(area_of_interest[image_line]) - 1) else 0
                    g_i_minus_one = area_of_interest[image_line][pixel_value - 1] if pixel_value != 0 else 0
                    current_edgeness.append(abs(g_i_plus_one - g_i_minus_one) + abs(g_j_plus_one - g_j_minus_one))
            edgeness.append([np.mean(current_edgeness)])
        return edgeness

    # Used for human recognition
    def get_human_features(self):
        humans = []
        for image in self.images:
            blur = np.array(cv2.bilateralFilter(image, 6, 75, 75))  # Filters image

            # Converts image to HSV
            original = matplotlib.colors.rgb_to_hsv(image)
            blurred = matplotlib.colors.rgb_to_hsv(blur)

            humans.append([np.mean(original - blurred)])  # Subtracts images and calculates mean
        return humans

    def get_age_features(self):
        growth_ratios = self.extract_growth_ratios()
        forehead_wrinkle_density = self.get_forehead_wrinkles()
        eye_wrinkle_density = self.mean_av_wrinkle_density()
        right_cheek_wrinkle_density = self.get_cheek_densities(self.get_right_eye())
        left_cheek_wrinkle_density = self.get_cheek_densities(self.get_left_eye())
        average_cheek_wrinkle_density = [[np.mean([right_cheek_wrinkle_density[i], left_cheek_wrinkle_density[i]])] for i in range(len(right_cheek_wrinkle_density))]

        for i in range(len(growth_ratios)):
            growth_ratios[i].append(forehead_wrinkle_density[i])
            growth_ratios[i].append(average_cheek_wrinkle_density[i][0])
            growth_ratios[i].append(eye_wrinkle_density[i])

        return growth_ratios

    def av_wrinkle_density(self, image, eye, distance_between_eyes, left=True):
        fact = -1 if left else 1
        com_calc = eye[0] + ((5 / 12) * fact) * distance_between_eyes
        com_calc2 = com_calc + ((1 / 6) * fact) * distance_between_eyes
        com_calc3 = eye[1] + (1 / 4) * distance_between_eyes

        top_left = (com_calc2 if left else com_calc, eye[1])
        bottom_right = (com_calc if left else com_calc2, com_calc3)

        wrinkle_area = image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]
        wrinkle_edges = cv2.Canny(wrinkle_area, 100, 200)
        return np.count_nonzero(wrinkle_edges) / (wrinkle_edges.shape[0] * wrinkle_edges.shape[1])


    def mean_av_wrinkle_density(self):
        average_cheek_wrinkle_density = []
        left_eyes = self.get_left_eye()
        right_eyes = self.get_right_eye()
        distance_between_eyes = self.get_distance_between_eyes()

        for i, val in enumerate(self.images):
            average_cheek_wrinkle_density.append(np.mean([self.av_wrinkle_density(val, left_eyes[i], distance_between_eyes[i], True), self.av_wrinkle_density(val, right_eyes[i], distance_between_eyes[i], False)]))

        return average_cheek_wrinkle_density

    def extract_growth_ratios(self):
        left_eyes = self.get_left_eye()
        right_eyes = self.get_right_eye()
        nose_tips = self.get_nose_tip()
        chins = self.get_chin()
        middle_of_eyes = self.get_middle_of_eyes()
        lips = self.get_lip()
        top_of_heads = self.get_top_of_head()
        right_face_side, left_face_side = self.get_face_sides()
        ratios = []

        for i in range(len(left_eyes)):

            # Calculate ratios
            # Ratio 1: D(left_eye, right_eye)/D(middle_of_eyes, nose)
            r1 = calculate_distance(left_eyes[i], right_eyes[i]) / calculate_distance(middle_of_eyes[i], nose_tips[i])

            # Ratio 2: D(left_eye, right_eye)/D(middle_of_eyes, lip)
            r2 = calculate_distance(left_eyes[i], right_eyes[i]) / calculate_distance(middle_of_eyes[i], lips[i])

            # Ratio 3: D(left_eye, right_eye)/D(middle_of_eyes, chin)
            r3 = calculate_distance(left_eyes[i], right_eyes[i]) / calculate_distance(middle_of_eyes[i], chins[i])

            # Ratio 4: D(middle_of_eyes, nose)/D(middle_of_eyes, lip)
            r4 = calculate_distance(middle_of_eyes[i], nose_tips[i]) / calculate_distance(middle_of_eyes[i], lips[i])

            # Ratio 5: D(middle_of_eyes, lip)/D(middle_of_eyes, chin)
            r5 = calculate_distance(middle_of_eyes[i], lips[i]) / calculate_distance(middle_of_eyes[i], chins[i])

            # Ratio 6: D(left_eye, right_eye)/D(top_of_head, chin)
            r6 = calculate_distance(middle_of_eyes[i], lips[i]) / calculate_distance(top_of_heads[i], chins[i])

            # Ratio 7: D(left_side, right_side)/D(top_of_head, chin)
            r7 = calculate_distance(right_face_side[i], left_face_side[i]) / calculate_distance(top_of_heads[i], chins[i])
            ratios.append([r1, r2, r3, r4, r5, r6, r7])

        return ratios

    def get_cheek_densities(self, eye):
        distance_between_eyes = self.get_distance_between_eyes()
        average_cheek_densities = []

        for i, val in enumerate(self.images):
            cheek_top_left = (eye[i][0] - (1 / 12) * distance_between_eyes[i], eye[i][1] + (1 / 8) * distance_between_eyes[i])
            cheek_bottom_right = (eye[i][0] + (1 / 12) * distance_between_eyes[i], eye[i][1] + (1 / 8) * distance_between_eyes[i] + (1 / 4) * distance_between_eyes[i])
            cheek_wrinkle_area = val[int(cheek_top_left[1]):int(cheek_bottom_right[1]), int(cheek_top_left[0]):int(cheek_bottom_right[0])]
            cheek_wrinkle_edges = cv2.Canny(cheek_wrinkle_area, 100, 200)
            average_cheek_densities.append(np.count_nonzero(cheek_wrinkle_edges) / (cheek_wrinkle_edges.shape[0] * cheek_wrinkle_edges.shape[1]))
        return average_cheek_densities

    # Forehead area wrinkles
    def get_forehead_wrinkles(self):
        forehead_wrinkles = []
        middle_of_eyes = self.get_middle_of_eyes()
        distance_between_eyes = self.get_distance_between_eyes()

        for i, val in enumerate(self.images):
            forehead_top_left = (middle_of_eyes[i][0] - (2 / 3) * distance_between_eyes[i], middle_of_eyes[i][1] - (distance_between_eyes[i] * 0.75) - (1 / 3) * distance_between_eyes[i])
            forehead_bottom_right = (middle_of_eyes[i][0] + (2 / 3) * distance_between_eyes[i], middle_of_eyes[i][1] - (distance_between_eyes[i] * 0.75))
            forehead_area = val[int(forehead_top_left[1]):int(forehead_bottom_right[1]), int(forehead_top_left[0]):int(forehead_bottom_right[0])]
            forehead_edges = cv2.Canny(forehead_area, 100, 200)
            forehead_wrinkles.append(np.count_nonzero(forehead_edges) / (forehead_edges.shape[0] * forehead_edges.shape[1]))
        return forehead_wrinkles

    def get_face_sides(self):
        chins = [landmark["chin"] for landmark in self.face_landmarks_list]
        lips = self.get_lip()
        right_side = []
        left_side = []
        for i in range(len(self.face_landmarks_list)):
            # Get coordinates for sides of face based on height of lip
            right_side_ordered = sorted([coords for coords in chins[i] if coords[0] > lips[i][0]], key=lambda x: abs(lips[0][1] - x[1]))
            left_side_ordered = sorted([coords for coords in chins[i] if coords[0] < lips[i][0]], key=lambda x: abs(lips[i][1] - x[1]))

            x_lip_line = np.array([list(lips[i]), [lips[i][0] + 1, lips[i][1]]])

            # Convert coordinates to numpy array
            right_side_coords = np.array([list(right_side_ordered[0]), list(right_side_ordered[1])])
            left_side_coords = np.array([list(left_side_ordered[0]), list(left_side_ordered[1])])

            # Find point on side of the face where the point at lip height crosses the sides of the face
            t1, s1 = np.linalg.solve(np.array([right_side_coords[1] - right_side_coords[0], x_lip_line[0] - x_lip_line[1]]).T, x_lip_line[0] - right_side_coords[0])
            t2, s2 = np.linalg.solve(np.array([left_side_coords[1] - left_side_coords[0], x_lip_line[0] - x_lip_line[1]]).T, x_lip_line[0] - left_side_coords[0])

            # Calculate intercepts
            right_side.append(list(map(int, (1 - t1) * right_side_coords[0] + t1 * right_side_coords[1])))
            left_side.append(list(map(int, (1 - t2) * left_side_coords[0] + t2 * left_side_coords[1])))

        return right_side, left_side
