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

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.preprocessing import StandardScaler

import sys

# check tf has access to gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print("GPUs available:", gpus)
else:
  print("No GPUs were found")

ID_NAME = ''
L_WEIGHTS = '/ghome/group07/task1-add_change_layers/PATCHES_32SIZE_5LAYERS/weights.h5'

print("ID_NAME is: {}".format(ID_NAME))
print("LOADED MODEL is: {}".format(L_WEIGHTS))

#user defined variables
IMG_SIZE = 256
PATCH_SIZE  = 32 # Before 64
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
#PATCHES_DIR = '/ghome/group07/work/C3/data/MIT_split_patches'+str(PATCH_SIZE)
PATCHES_DIR = '/ghome/group07/task1-add_change_layers/patches_dir/32'
#MODEL_FNAME = '/ghome/group07/work/C3/patch_based_mlp.weights.h5'

NUM_PATCHES = (IMG_SIZE//PATCH_SIZE)**2

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
  directory=DATASET_DIR+'/train/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(PATCH_SIZE, PATCH_SIZE)
)

# Load and preprocess the validation dataset
validation_dataset = keras.preprocessing.image_dataset_from_directory(
  directory=DATASET_DIR+'/test/',
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

#--------------------------------------------------
# LOAD MODEL
#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Input(shape=(PATCH_SIZE, PATCH_SIZE, 3,),name='input'))
model.add(Reshape((PATCH_SIZE*PATCH_SIZE*3,)))
model.add(Dense(units=2048, activation='relu', name='first'))
model.add(Dense(units=1024, activation='relu', name='second'))
model.add(Dense(units=512, activation='relu',name='third'))
model.add(Dense(units=256, activation='relu',name='fourth'))
model.add(Dense(units=128, activation='relu',name='fifth'))
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

directory = DATASET_DIR+'/test'
train_directory = DATASET_DIR+'/train'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
correct = 0.
total   = 807
count   = 0

img_patches = []
train_labels = []
for class_dir in os.listdir(train_directory):
  cls = classes[class_dir]
  for imname in os.listdir(os.path.join(train_directory,class_dir)):
    train_labels.append(cls)
    im = Image.open(os.path.join(train_directory,class_dir,imname))
    patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=(int(256/PATCH_SIZE)**2))
    img_patches.append(patches)

img_patches_val = []
val_labels = []
for class_dir in os.listdir(directory):
  cls = classes[class_dir]
  for imname in os.listdir(os.path.join(directory,class_dir)):
    val_labels.append(cls)
    im = Image.open(os.path.join(directory,class_dir,imname))
    patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=(int(256/PATCH_SIZE)**2))
    img_patches_val.append(patches)

def SVM_BoW_layer(layer_model, optimal_k):

  layer_model.summary()
  
  features_train = np.empty((len(img_patches), NUM_PATCHES, layer_model.layers[-1].output_shape[1]))
  for i in range(len(img_patches)):
    image = img_patches[i]
    # print(image.shape)
    features_train[i, :, :] = layer_model.predict(image/255., verbose=0)

  features_val = np.empty((len(img_patches_val), NUM_PATCHES, layer_model.layers[-1].output_shape[1]))
  for i in range(len(img_patches_val)):
    image = img_patches_val[i]
    features_val[i, :, :] = layer_model.predict(image/255., verbose=0)

  print("features train shape:", features_train.shape)

  #k=198 # obtained in week 1
  k=optimal_k # new (optuna optimized) = # units * 3.05
  codebook = MiniBatchKMeans(n_init=3, n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
  print('Codebook created')
  codebook.fit(np.vstack(features_train))
  print('Codebook fitted')
  
  visual_words = StandardScaler().fit_transform(predictBoVW(codebook, features_train, k=k))
  print('Visual words predicted')

  visual_words_val = StandardScaler().fit_transform(predictBoVW(codebook, features_val, k=k))
  print('Visual words validation predicted\n\n')

  SVM = svm.SVC(kernel='rbf')
  SVM.fit(visual_words, train_labels)
  print('SVM fitted')
  print(f'SVM accuracy: {SVM.score(visual_words_val, val_labels)}')
  # print(f'SVM accuracy: {SVM.score(visual_words_val, np.argmax(svm_val_y, axis=1))}\n\n\n')


i=1
units_arr = {"1": 2048, "2": 1024, "3": 512, "4": 256, "5": 128}
for layer_mod in model_layers:
  optimal_k = 621
  print(f'CREATING BoW AND USING IT AS INPUT FOR SVM, WITH FEATURES AS OUTPUTS OF HIDDEN LAYER{i},\n the name of the layer is: {layer_names[i]}\t k = {optimal_k}\n')
  i+=1

  SVM_BoW_layer(layer_mod, optimal_k)


print('Done!\n')
#print('Test Acc. = '+str(correct/total)+'\n')