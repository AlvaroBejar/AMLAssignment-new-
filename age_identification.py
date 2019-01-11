from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import helper_functions
from joblib import dump


def SVM(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='linear', probability=True)  # Creates SVM to train
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    print(str(accuracy_score(y_test, predicted)))
    return clf

# Code used if features want to be extracted live
'''
images = helper_functions.load_data("all_images.txt")
fe = FeatureExtractor(images)
features = fe.get_multiple_features(["age_features"])[0]
'''

# Load features and labels
features = helper_functions.load_data("saved_features/age_identification_features.txt")
y = helper_functions.get_labels_for_task("young")

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3)

mod = SVM(x_train, x_test, y_train, y_test)

dump(mod, "models/age_identification_model.joblib")  # Save model


'''
class_names = ["Young", "Old"]
cnf_matrix = confusion_matrix(y_test, predicted2)
data_plotting.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.show()
'''
