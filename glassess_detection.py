from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import helper_functions
from Features import FeatureExtractor
from joblib import dump
import numpy as np

# Logistic regressor with set parameters
def logistic_regressor(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression(C=1, solver='liblinear', multi_class='ovr')  # Creates regressor
    logreg.fit(x_train, y_train)  # Train model
    predicted = logreg.predict(x_test)
    print(str(accuracy_score(y_test, predicted)))
    return logreg

# Code used if features want to be extracted live
'''
data = helper_functions.load_data("all_images.txt")
fe = FeatureExtractor(data)
edgeness = fe.get_multiple_features(["edgeness"])[0]
'''

# Load saved features and labels
edgeness = helper_functions.load_data("saved_features/glasses_detection_features.txt")
y = helper_functions.get_labels_for_task("eyeglasses")

# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(edgeness, y, test_size=0.3)

log = logistic_regressor(x_train, y_train, x_test, y_test)
dump(log, "models/glasses_detection_model.joblib")  # Save model
