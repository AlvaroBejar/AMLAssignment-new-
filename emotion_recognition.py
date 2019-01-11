from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import helper_functions
from joblib import dump

# Code used if features want to be extracted live

'''
file_dir = ["dataset/", ".png"]
images = []
for file_no in constants.images:
    file_name = str(file_no).join(file_dir)
    images.append(face_recognition.load_image_file(file_name))

fe = FeatureExtractor(images)
data = fe.get_multiple_features(["mouth"])
'''

def SVM(x_train_flattened, y_train, x_test_flattened):
    clf = svm.SVC(kernel='linear', probability=True)  # Creates SVM to train
    clf.fit(x_train_flattened, y_train)  # Fits model to data
    predicted = clf.predict(x_test_flattened)
    print(accuracy_score(y_test, predicted))
    return clf

# Loads saved features and labels
data = helper_functions.load_data("saved_features/emotion_recognition_features.txt")[0]
y = helper_functions.get_labels_for_task("smiling")

# Split data to train and test
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3)

# Reformats data from 3d to 2d for x_train and x_test
nsamples_train, nx_train, ny_train = x_train.shape
x_train_flattened = x_train.reshape((nsamples_train, nx_train * ny_train))

nsamples_test, nx_test, ny_test = x_test.shape
x_test_flattened = x_test.reshape((nsamples_test, nx_test * ny_test))

mod = SVM(x_train_flattened, y_train, x_test_flattened)  # Train model

dump(mod, "emotion_recognition_model.joblib")  # Save model
