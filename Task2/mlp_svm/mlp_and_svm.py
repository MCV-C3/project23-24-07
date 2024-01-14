import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model

from sklearn import svm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import sys

ID_NAME = sys.argv[1]
L_WEIGHTS = sys.argv[2]

print("ID_NAME is: {}".format(ID_NAME))
print("LOADED MODEL is: {}".format(L_WEIGHTS))

SAVE_DIR = os.path.join(os.path.dirname(__file__), ID_NAME)
#os.mkdir(SAVE_DIR)

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
#MODEL_FNAME = os.path.join(SAVE_DIR, 'weights.h5') # MODIFIED

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()


print('Setting up data ...\n')

#--------------------
classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']
#--------------------

# Load and preprocess the training dataset
train_dataset = keras.utils.image_dataset_from_directory(
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
validation_dataset = keras.utils.image_dataset_from_directory(
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
preprocessing_train = keras.Sequential([
  keras.layers.Rescaling(1./255),
  keras.layers.RandomFlip("horizontal")
])

preprocessing_validation = keras.Sequential([
  keras.layers.Rescaling(1./255)
])


train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))


train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#--------------------------------------------------
# LOAD MODEL
#Build the Multi Layer Perceptron model
model = Sequential()
input = Input(shape=(IMG_SIZE, IMG_SIZE, 3,),name='input')
model.add(input) # Input tensor
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),name='reshape'))
model.add(Dense(units=2048, activation='relu',name='first'))
# model.add(Dense(units=1024, activation='relu',name='second'))
model.add(Dense(units=8, activation='softmax',name='classification'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


print(model.summary())
#plot_model(model, to_file=os.path.join(SAVE_DIR, 'modelMLP.png'), show_shapes=True, show_layer_names=True) # MODIFIED

model.load_weights(L_WEIGHTS, skip_mismatch=False)
layer_names = [layer.name for layer in model.layers]
print(f'Layer names:\n{layer_names}\n')
print('We will work separately with SVMs for each hidden layer\n\n')

# HERE WE HAVE TO CHOSE THE LAYER TO GET THE FEATURES FOR THE SVM
# WE MUST CHOOSE ONE OF THE DENSE LAYERS, THE FIRST HIDDEN
# LAYER'S INDEX IS 1
model_layers = []

for i in range(1, len(layer_names)-1):
  mod = keras.Model(inputs=model.input, outputs=model.get_layer(layer_names[i]).output)
  model_layers.append(mod)
#--------------------------------------------------



def SVM_layer(layer_model):

  layer_model.summary()
  n_features = layer_model.output_shape[-1]

  #layer_name = layer_model.get_layer(index=-1).name
  
  svm_train_x = np.array([])
  svm_train_y = np.array([])
  for x, y in train_dataset:
    #svm_train_x.append(layer_model.predict(x, verbose=0), axis=1)
    #svm_train_y.append(y, axis=1)
    svm_train_x = np.append(svm_train_x, layer_model.predict(x, verbose=0))
    #print('svm_train_x', len(svm_train_x), svm_train_x)
    svm_train_y = np.append(svm_train_y, y)
    #print('svm_train_y', len(svm_train_y), svm_train_y)
  #svm_train_x = np.asarray(svm_train_x)
  #svm_train_y = np.asarray(svm_train_y)

  svm_train_x = np.reshape(svm_train_x, (-1, n_features))
  svm_train_y = np.reshape(svm_train_y, (-1, 8))

  

  svm_val_x = np.array([])
  svm_val_y = np.array([])
  for x, y in validation_dataset:
    #svm_val_x.append(layer_model.predict(x, verbose=0), axis=1)
    #svm_val_y.append(y, axis=1)
    svm_val_x = np.append(svm_val_x, layer_model.predict(x, verbose=0))
    svm_val_y = np.append(svm_val_y, y)
  #svm_val_x = np.asarray(svm_val_x)
  #svm_val_y = np.asarray(svm_val_y)

  svm_val_x = np.reshape(svm_val_x, (-1, n_features))
  svm_val_y = np.reshape(svm_val_y, (-1, 8))


  SVM = svm.SVC(kernel='rbf')
  SVM.fit(svm_train_x, np.argmax(svm_train_y, axis=1))

  #print(f'SVM accuracy: {SVM.score(svm_train_x, np.argmax(svm_train_y, axis=1))}\n\n\n')
  print(f'SVM accuracy: {SVM.score(svm_val_x, np.argmax(svm_val_y, axis=1))}\n\n\n')


i=1
for layer_mod in model_layers:
  print(f'TRAINING SVM WITH FEATURES AS OUTPUTS OF HIDDEN LAYER{i},\n the name of the layer is: {layer_names[i]}\n')
  i+=1

  SVM_layer(layer_mod)


print('Done!')