import tensorflow as tf
from tensorflow import keras
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

#import images
import cv2
import imghdr
data_dir = 'data' 
image_exts = ['.png']

#cleaning images
for image_class in os.listdir(data_dir):
   for image in os.listdir(os.path.join(data_dir, image_class)):
      image_path = os.path.join(data_dir, image_class, image)
      try:
         img = cv2.imread(image_path)
         tip = imghdr.what(image_path)
         if tip not in image_exts:
            print('Image not in ext list {}'.format(image_path))
            os.remove(image_path)
      except Exception as e:
         print('Issue with image{}' .format(image_path))
         # os.remove(image_path)

#load data
data = tf.keras.utils.image_dataset_from_directory('data', shuffle = False)
data = data.shuffle(1000, seed=100, reshuffle_each_iteration=False)
tf.keras.utils.image_dataset_from_directory('data', batch_size = 50, image_size=(230,230))
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols = 4, figsize = (20,20))
for idx, image in enumerate(batch[0][:4]):
    ax[idx].imshow(image.astype(int))
    ax[idx].title.set_text(batch[1][idx])

#scale data (?)
data = data.map(lambda x, y: (x/255, y))
data.as_numpy_iterator().next()

#split data into training and testing (maybe not needed later)
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

#implement the model
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

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

print(pre.result(), re.result(), acc.result())


#how to test an image:

img = cv2.imread('154006829.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))


#Binary classification (UPDATE FOR MULTICLASSIFICATION)
if yhat > 0.5:
    print('Sad')
else:
   print('Happy')


#Save the model
from keras.api.models import load_model
model.save(os.path.join('models', 'model.h5'))
new_model = load_model('model.h5')
new_model.predict(np.expand_dims(resize/255, 0))

