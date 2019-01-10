from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import helper_functions
from Features import FeatureExtractor
from joblib import dump

# Logistic regressor with set parameters
def logistic_regressor(x_train, y_train, x_test, y_test):
    logreg = LogisticRegression(C=1, solver='liblinear', multi_class='ovr')
    logreg.fit(x_train, y_train)
    predicted = logreg.predict(x_test)
    print(str(accuracy_score(y_test, predicted)))
    return logreg

'''
data = helper_functions.load_data("all_images.txt")
fe = FeatureExtractor(data)
edgeness = fe.get_multiple_features(["edgeness"])[0]
'''

edgeness = helper_functions.load_data("saved_features/glasses_detection.txt")
y = helper_functions.get_labels_for_task("eyeglasses")
x_train, x_test, y_train, y_test = train_test_split(edgeness, y, test_size=0.3)

log = logistic_regressor(x_train, y_train, x_test, y_test)
dump(log, "glasses_detection_model.joblib")

'''
sorted_colors = [z for z, x in sorted(zip(y,edgeness))]
plt.scatter([i for i in range(len(constants.images))], sorted(edgeness), c=sorted_colors)
plt.show()
'''



