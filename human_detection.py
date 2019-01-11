from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import helper_functions
from Features import FeatureExtractor
from joblib import dump

# Logistic regressor with set parameters
def logistic_regressor(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression(C=1, solver='liblinear', multi_class='ovr')  # Create model
    logreg.fit(x_train, y_train)  # Fit model
    predicted = logreg.predict(x_test)
    print(str(accuracy_score(y_test, predicted)))
    return logreg

# Code used if features want to be extracted live
'''
data = helper_functions.load_data("all_images.txt")
fe = FeatureExtractor(data)
human = fe.get_multiple_features(["human_values"])[0]
'''

# Get features and labels
human = helper_functions.load_data("saved_features/human_detection_features.txt")
y = helper_functions.get_labels_for_task("human")

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(human, y, test_size=0.3)
log = logistic_regressor(x_train, y_train, x_test, y_test)  # Train model

dump(log, "human_detection_model.joblib")  # Save model
