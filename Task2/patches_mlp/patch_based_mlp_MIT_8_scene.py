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
PATCH_SIZE = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
PATCHES_DIR = os.path.join(SAVE_DIR, 'patches_dir',str(PATCH_SIZE)) # MODIFIED
MODEL_FNAME = os.path.join(SAVE_DIR, 'weights.h5') # MODIFIED


if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()
if not os.path.exists(PATCHES_DIR):
  print('WARNING: patches dataset directory '+PATCHES_DIR+' does not exist!\n')
  print('Creating image patches dataset into '+PATCHES_DIR+'\n')
  generate_image_patches_db(DATASET_DIR,PATCHES_DIR,patch_size=PATCH_SIZE)
  print('patxes generated!\n')

# Data augmentation and preprocessing
preprocessing_train = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
])

preprocessing_validation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

# Load and preprocess the training dataset
train_dataset = keras.preprocessing.image_dataset_from_directory(
  directory=PATCHES_DIR+'/train/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(PATCH_SIZE, PATCH_SIZE)
)

# Load and preprocess the validation dataset
validation_dataset = keras.preprocessing.image_dataset_from_directory(
  directory=PATCHES_DIR+'/test/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(PATCH_SIZE, PATCH_SIZE)
)

train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



def build_mlp(input_size=PATCH_SIZE,phase='train'):
  model = Sequential()
  model.add(Input(shape=(input_size, input_size, 3,),name='input'))
  model.add(Reshape((input_size*input_size*3,)))
  model.add(Dense(units=512, activation='relu'))
  model.add(Dense(units=256, activation='relu'))
  model.add(Dense(units=128, activation='relu'))
  #model.add(Dense(units=256, activation='relu'))
  #model.add(Dense(units=128, activation='relu'))
  if phase=='test':
    model.add(Dense(units=8, activation='linear')) # In test phase we softmax the average output over the image patches
  else:
    model.add(Dense(units=8, activation='softmax'))
  return model


print('Building MLP model...\n')

model = build_mlp(input_size=PATCH_SIZE)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

train = False
if  not os.path.exists(MODEL_FNAME) or train:
  print('WARNING: model file '+MODEL_FNAME+' do not exists!\n')
  print('Start training...\n')
  
  model.fit(train_dataset,
            epochs=150,
            validation_data=validation_dataset,
            verbose=0)
  
  print('Saving the model into '+MODEL_FNAME+' \n')
  model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
  print('Done!\n')

  # print('Accuracy of the model on the sets:')
  # print(f'Train set: {history.history["accuracy"][-1]}')
  # print(f'Validation set: {history.history["val_accuracy"][-1]}\n')

  # print('Loss of the model on the sets:')
  # print(f'Train set: {history.history["loss"][-1]}')
  # print(f'Validation set: {history.history["val_loss"][-1]}')

print('Building MLP model for testing...\n')

model = build_mlp(input_size=PATCH_SIZE, phase='test')
print(model.summary())

print('Done!\n')

print('Loading weights from '+MODEL_FNAME+' ...\n')
print ('\n')

model.load_weights(MODEL_FNAME)

print('Done!\n')

print('Start evaluation ...\n')

directory = DATASET_DIR+'/test'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
correct = 0.
total   = 807
count   = 0

for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory,class_dir)):
      im = Image.open(os.path.join(directory,class_dir,imname))
      patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=(int(256/PATCH_SIZE)**2))
      out = model.predict(patches/255.)
      predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )
      if predicted_cls == cls:
        correct+=1
      count += 1
      print('Evaluated images: '+str(count)+' / '+str(total), end='\r')

print('Done!\n')
print('Test Acc. = '+str(correct/total)+'\n')
