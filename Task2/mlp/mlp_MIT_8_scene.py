import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import sys

# check tf has access to gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print("GPUs available:", gpus)
else:
  print("No GPUs were found")

ID_NAME = sys.argv[1] if len(sys.argv) > 2 else ''

print("ID_NAME is: {}".format(ID_NAME))

SAVE_DIR = os.path.join(os.path.dirname(__file__), ID_NAME)
#os.mkdir(SAVE_DIR)

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
MODEL_FNAME = os.path.join(SAVE_DIR, 'weights.h5') # MODIFIED

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()


print('Setting up data ...\n')

#--------------------
classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
#--------------------

# Load and preprocess the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  directory=DATASET_DIR+'/train/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(IMG_SIZE, IMG_SIZE),
  shuffle=True,
  validation_split=None,
  subset=None
)

# Load and preprocess the validation dataset
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  directory=DATASET_DIR+'/test/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(IMG_SIZE, IMG_SIZE),
  shuffle=True,
  seed=123,
  validation_split=None,
  subset=None
)

# Data augmentation and preprocessing
preprocessing_train = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
])

preprocessing_validation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


print('Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
input = Input(shape=(IMG_SIZE, IMG_SIZE, 3,),name='input')
model.add(input) # Input tensor
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),name='reshape'))

model.add(Dense(units=2048, activation='relu', name='first'))
model.add(Dense(units=1024, activation='relu', name='second'))
model.add(Dense(units=512, activation='relu', name='third'))
# model.add(Dense(units=256, activation='relu', name='forth'))
# model.add(Dense(units=128, activation='relu', name='fifth'))
model.add(Dense(units=8, activation='softmax', name='classification'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file=os.path.join(SAVE_DIR, 'modelMLP.png'), show_shapes=True, show_layer_names=True) # MODIFIED

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

#----------------------------------------------------
# Star computing the training time
start_time = start_training_timer()
#----------------------------------------------------


print('Start training...\n')
history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        verbose=0)

#----------------------------------------------------
write_time(start_time)
#----------------------------------------------------

print('Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(MODEL_FNAME)  # always save your weights after training or during training

  # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(os.path.join(SAVE_DIR,'accuracy.jpg'))
plt.close()

  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(os.path.join(SAVE_DIR,'loss.jpg'))

#to get the output of a given layer
 #crop the model up to a certain layer
layer = 'first'
model_layer = keras.Model(inputs=input, outputs=model.get_layer(layer).output)


#get the features from images
directory = DATASET_DIR+'/test/coast'
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
print(f'prediction for image {os.path.join(directory, os.listdir(directory)[0] )} on  layer {layer}')
features = model_layer.predict(x/255.0)
#print(f'The first layer predicted this element as {classes[np.argmax(features)]}')
#print(f'and the computed probability is: {np.max(features)}')
#real_class = directory.split('/')[-1]
#print(f'The element belongs in reality to class {real_class}\n')
print(features.shape)
print(features)

#get classification
classification = model.predict(x/255.0)
print(f'classification for image {os.path.join(directory, os.listdir(directory)[0] )}:')
classification = classification/np.sum(classification,axis=1)
classification = np.round(classification * 100, 2)
print(f'The model predicted this element as {classes[np.argmax(classification)]}')
print(f'and the computed probability is: {np.max(classification)}')
real_class = directory.split('/')[-1]
print(f'The element belongs in reality to class {real_class}\n\n')
print(classification)

print('Accuracy of the model on the sets:')
print(f'Train set: {history.history["accuracy"][-1]}')
print(f'Validation set: {history.history["val_accuracy"][-1]}\n')

print('Loss of the model on the sets:')
print(f'Train set: {history.history["loss"][-1]}')
print(f'Validation set: {history.history["val_loss"][-1]}')

print('Done!')