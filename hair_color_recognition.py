import keras
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras import backend as K
import helper_functions

model_accuracy = []
images = helper_functions.load_data("all_images.txt")
y =helper_functions.get_labels_for_task("hair_color")
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.3)

# Converts labels to categorical for multi-class classification
num_classes = 7
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

epochs = 5
batch_size = 16

img_width, img_height = 256, 256

# Set input format consistent with model
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Create model with layers
model = Sequential()
model.add(Conv2D(16, (7, 7), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(7, activation='sigmoid'))

model.summary()

# Compiles model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fits model to data
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)


model.save('hair_detection_model.h5')  # creates a HDF5 file 'my_model.h5'

model_accuracy = []
score = model.evaluate(x_test, y_test, verbose=0)
model_accuracy.append(score[1])
print('Test loss:', score[0])
print('Test accuracy:', score[1])

