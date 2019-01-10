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
human = fe.get_multiple_features(["human_values"])[0]
'''

human = helper_functions.load_data("saved_features/human_detection_features.txt")
helper_functions.save_data("saved_features/human_detection_features.txt", human)

y = helper_functions.get_labels_for_task("human")
x_train, x_test, y_train, y_test = train_test_split(human, y, test_size=0.3)
log = logistic_regressor(x_train, y_train, x_test, y_test)

dump(log, "human_detection_model.joblib")


'''
predicted = log.predict_proba(var)
predicted = [[i + 1, val[1]] for i, val in enumerate(predicted)]
print(predicted)

with open("human_detection_C.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(predicted)
    
class_names = ["Glasses", "No glasses"]
cnf_matrix = confusion_matrix(y_test, predicted)
data_plotting.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.show()
'''
