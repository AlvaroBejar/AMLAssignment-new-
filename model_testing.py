from keras.models import load_model
from joblib import load
from os import listdir
from os.path import isfile, join
import face_recognition
from Features import FeatureExtractor
import numpy as np
import csv

# Saves variables to csv
def save_to_csv(file_name, values):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerows(values)

# Test model with data and save results to csv
def save_results_to_csv(model, features, file_name):

    if features == "mouth":
        feature = np.array(fe.get_multiple_features([features])[0])

        # Reformats data from 3d to 2d for x_train and x_test
        nsamples, nx, ny = feature.shape
        feature = feature.reshape((nsamples, nx * ny))
    else:
        feature = fe.get_multiple_features([features])[0]

    predicted = model.predict_proba(feature)  # Gets probabilities for testing data
    values = sorted([[indexes[i], val[1]] for i, val in enumerate(predicted)])  # Gets probability of task
    save_to_csv(file_name, values)

# Load all models
emotion_recognition_model = load('models/emotion_recognition_model.joblib')
age_identification_model = load('models/age_identification_model.joblib')
glasses_detection_model = load('models/glasses_detection_model.joblib')
human_detection_model = load('models/human_detection_model.joblib')
hair_detection_model = load_model('models/hair_detection_model.h5')

# Read files in directory and drop unwanted files
file_list = [join("testing_dataset/", f) for f in listdir("testing_dataset/") if isfile(join("testing_dataset/", f))]
file_list.remove("testing_dataset/.DS_Store")
images = []
indexes = []
for file in file_list:  # Reads images and saves them
    images.append(face_recognition.load_image_file(file))
    indexes.append(int(file.split("/")[1].split(".")[0]))

fe = FeatureExtractor(images)

np.set_printoptions(suppress=True)

# Test models on training data and save results to csv
save_results_to_csv(emotion_recognition_model, "mouth", "test_results/emotion_recognition_C.csv")
save_results_to_csv(age_identification_model, "age_features", "test_results/age_identification_C.csv")
save_results_to_csv(glasses_detection_model, "edgeness", "test_results/glasses_detection_C.csv")
save_results_to_csv(human_detection_model, "human_values", "test_results/human_detection_C.csv")
pred = hair_detection_model.predict(np.array(images))
save_to_csv("test_results/hair_color_recognition_C", pred)