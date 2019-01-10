import constants
import face_recognition
import glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
from PIL import Image

import time
start_time = time.time()

def find_outliers(path):
    outliers = []
    for image_path in glob.glob(path):
        print(image_path)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            outliers.append(int(image_path.split("/")[1].split(".png")[0]))

    return outliers


# From sklearn
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

path = "dataset/*.png"
nature_images = find_outliers(path)

y_true = [0 if i in constants.faces else 1 for i in range(1, 5001)]
y_pred = [1 if i in constants.outliers else 0 for i in range(1, 5001)]

class_names = ["Face images", "Outliers"]
cnf_matrix = confusion_matrix(y_true, y_pred)
print("--- %s seconds ---" % (time.time() - start_time))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

'''
file_dir = ["dataset/", ".png"]
for file_no in range(1, 5001):
    file_name = str(file_no).join(file_dir)
    image = face_recognition.load_image_file(file_name)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0 and file_no in constants.faces:
        print(file_no)
'''
