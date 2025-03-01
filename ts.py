import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

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

#divide all color vals by max to normalize 0-1 (for more custom data)
# img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
# )

#GLOBAL VARS
num_classes = 7
BATCH_SIZE = 16
IMAGE_SIZE = (730, 730)
EPOCH_COUNT = 10
IMAGE_SHAPE = (730, 730, 3)

#load data from directory
train = tf.keras.utils.image_dataset_from_directory('clean_images', seed = 123, validation_split = 0.2, subset="training", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, color_mode='rgb')
val = tf.keras.utils.image_dataset_from_directory('clean_images', seed = 123, validation_split = 0.2, subset="validation", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, color_mode='rgb')
data_iterator = train.as_numpy_iterator()
batch = data_iterator.next()


#one hot encode labels for ML
def one_hot_encode(image, label):
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=num_classes) 
    return image, label

data = train.map(one_hot_encode)
val = val.map(one_hot_encode)

#pixel normalization (0-1)
#normalization_layer = tf.keras.layers.Rescaling(1./255)
#data = data.map(lambda x, y: (normalization_layer(x), y))
#val = val.map(lambda x, y: (normalization_layer(x), y))

# Shuffle and split data into train, validation, and test sets (DEPRECATED)
#data = data.shuffle(1000, seed=100, reshuffle_each_iteration=False)
# cardinality = tf.data.experimental.cardinality(data).numpy()
# train_size = int(cardinality * 0.8)
# val_size = int(cardinality * 0.2)
# train = data.take(train_size)
# val = data.skip(train_size).take(val_size)

#train set collection (more custom)
# train = tf.data.Dataset.from_generator(
#     lambda: img_gen.flow_from_directory('Images', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse'),
#     output_signature=(tf.TensorSpec(shape=(BATCH_SIZE, 880, 500, 3), dtype=tf.float32), tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.float32))
# )
#One-Hot Encoding
#train = train.map(one_hot_encode)

#check shape of data
for image, label in train.take(1):
    print(f"Image shape: {image.shape}, Label shape: {label.shape}")

#implement the model
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Rescaling, InputLayer, RandomRotation, RandomZoom

#augmentation layer in the model (decrease overmatch)
augment = Sequential()
augment.add(RandomRotation(0.1))
augment.add(RandomZoom(0.1))

#Full DNN Keras model
model = Sequential()

model.add(InputLayer(shape=IMAGE_SHAPE))
model.add(augment)
model.add(Rescaling(1./255))
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
#model.add(Conv2D(32, (3,3), 1, activation='relu'))
#model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#model training
logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
#progbar_logger = tf.keras.callbacks.ProgbarLogger(count_mode="steps")
#hist = model.fit(train, epochs = 10, validation_data=val, callbacks=[tensorboard_callback])
history = model.fit(data, epochs=EPOCH_COUNT, validation_data=val, callbacks=[tensorboard_callback])

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(val, verbose=2)

#evaluation of model
# from keras.api.metrics import Precision, Recall, Accuracy

# pre = Precision()
# re = Recall()
# acc = Accuracy()

# for batch in test.as_numpy_iterator():
#     x, y = batch
#     y_pred = model.predict(x)
#     pre.update_state(y, y_pred)
#     re.update_state(y, y_pred)
#     acc.update_state(y, y_pred)

# print(pre.result().numpy(), re.result().numpy(), acc.result().numpy())

#Classifier
# yhat = model.predict(np.expand_dims(resize / 255, 0))
# predicted_class = np.argmax(yhat, axis=1)
# print(f'Predicted class: {predicted_class}') #will print one-hot encoded class


#Save the model
#from keras.api.models import load_model

model.save(os.path.join('modelv0.8.1.keras'))
# new_model = load_model('model.h5')
# new_yhat = new_model.predict(np.expand_dims(resize / 255, 0))
# new_predicted_class = np.argmax(new_yhat, axis=1)
# print(f'Predicted class with loaded model: {new_predicted_class}')

#TODO: optimize output display (one-hot to string) and model saving logic