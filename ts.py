import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.api.utils import to_categorical

print("TensorFlow version:", tf.__version__)

#signify if using GPU or not
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

#comment out when running without GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
   tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

#import image processing libraries
import cv2
import imghdr

#image augmentation for training
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
)

#specify number of classes
num_classes = 16
BATCH_SIZE = 64

#load data
data = tf.keras.utils.image_dataset_from_directory('Images', shuffle=True, batch_size=BATCH_SIZE, image_size=(730, 730))
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#image normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)
data = data.map(lambda x, y: (normalization_layer(x), y))

#one hot encode labels for ML
def one_hot_encode(image, label):
    label = tf.one_hot(label, depth=num_classes) 
    return image, label

data = data.map(one_hot_encode)

# Shuffle and split data into train, validation, and test sets
data = data.shuffle(1000, seed=100, reshuffle_each_iteration=False)
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

#image augmentation
train = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory('Images', batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse'),
    output_types=(tf.float32, tf.float32))

#implement the model
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

#model used
model = Sequential()

#model layers
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(730,730,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.add(Dropout(0.2))

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#model training
logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
hist = model.fit(train, epochs = 10, validation_data=val, callbacks=[tensorboard_callback])

#performance plotting section (disabled for testing)
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

#evaluation of model
from keras.api.metrics import Precision, Recall, Accuracy

pre = Precision()
re = Recall()
acc = Accuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    y_pred = model.predict(x)
    pre.update_state(y, y_pred)
    re.update_state(y, y_pred)
    acc.update_state(y, y_pred)

print(pre.result().numpy(), re.result().numpy(), acc.result().numpy())

#how to test an image:

img = cv2.imread('154006829.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))


#Classifier
yhat = model.predict(np.expand_dims(resize / 255, 0))
predicted_class = np.argmax(yhat, axis=1)
print(f'Predicted class: {predicted_class}') #will print one-hot encoded class


#Save the model
from keras.api.models import load_model

model.save(os.path.join('models', 'model.h5'))
new_model = load_model('model.h5')
new_yhat = new_model.predict(np.expand_dims(resize / 255, 0))
new_predicted_class = np.argmax(new_yhat, axis=1)
print(f'Predicted class with loaded model: {new_predicted_class}')

#TODO: optimize output display (one-hot to string) and model saving logic