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
import optuna

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

import sys

# check tf has access to gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print("GPUs available:", gpus)
else:
  print("No GPUs were found")

L_WEIGHTS = '/ghome/group07/task1-add_change_layers/PATCHES_32SIZE_1LAYERS/weights.h5'

#user defined variables
IMG_SIZE = 256
PATCH_SIZE  = 32 # Before 64
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
#PATCHES_DIR = '/ghome/group07/work/C3/data/MIT_split_patches'+str(PATCH_SIZE)
PATCHES_DIR = '/ghome/group07/task1-add_change_layers/patches_dir/32'
#MODEL_FNAME = '/ghome/group07/work/C3/patch_based_mlp.weights.h5'

NUM_PATCHES = (IMG_SIZE//PATCH_SIZE)**2

codebook_size = 512 # 760


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
#model.add(Dense(units=1024, activation='relu', name='second'))
#model.add(Dense(units=512, activation='relu',name='third'))
#model.add(Dense(units=256, activation='relu',name='fourth'))
#model.add(Dense(units=128, activation='relu',name='fifth'))
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
model_0 = model_layers[0]


directory = DATASET_DIR+'/test'
train_directory = DATASET_DIR+'/train'
classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
correct = 0.
total   = 807
count   = 0

img_patches = []
img_names_train = []
train_labels = []
for class_dir in os.listdir(train_directory):
  cls = classes[class_dir]
  for imname in os.listdir(os.path.join(train_directory,class_dir)):
    train_labels.append(cls)
    img_names_train.append(os.path.join(train_directory,class_dir,imname))
    im = Image.open(os.path.join(train_directory,class_dir,imname))
    patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=(int(256/PATCH_SIZE)**2))
    img_patches.append(patches)

img_patches_val = []
img_names_val = []
val_labels = []
for class_dir in os.listdir(directory):
  cls = classes[class_dir]
  for imname in os.listdir(os.path.join(directory,class_dir)):
    val_labels.append(cls)
    img_names_val.append(os.path.join(directory,class_dir,imname))
    im = Image.open(os.path.join(directory,class_dir,imname))
    patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=(int(256/PATCH_SIZE)**2))
    img_patches_val.append(patches)


Train_descriptors = []
Train_label_per_descriptor = []

def get_descriptors(model, images_filenames):
    descriptors = np.empty((len(images_filenames), NUM_PATCHES, model.layers[-1].output_shape[1]))
    for i,filename in enumerate(images_filenames):
        img = Image.open(filename)
        patches = image.extract_patches_2d(np.array(img), (PATCH_SIZE, PATCH_SIZE), max_patches=NUM_PATCHES)
        descriptors[i, :, :] = model.predict(patches/255.)

    return descriptors

def get_visual_words(descriptors, codebook, codebook_size):
    visual_words=np.empty((len(descriptors),codebook_size),dtype=np.float32)
    for i,des in enumerate(descriptors):
        words=codebook.predict(des)
        visual_words[i,:]=np.bincount(words,minlength=codebook_size)

    return StandardScaler().fit_transform(visual_words)

train_descriptors = get_descriptors(model_0, img_names_train)

def objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 25, 700)

    codebook = MiniBatchKMeans(n_clusters=n_clusters,
                               batch_size=n_clusters * 20,
                               compute_labels=False,
                               reassignment_ratio=10**-4,
                               random_state=42)
    codebook.fit(np.vstack(train_descriptors))
    
    train_visual_words = get_visual_words(train_descriptors, codebook, n_clusters)

    # test
    test_descriptors = get_descriptors(model_0, img_names_val)
    test_visual_words = get_visual_words(test_descriptors, codebook, n_clusters)

    classifier = svm.SVC(kernel='rbf')
    classifier.fit(train_visual_words, train_labels)

    #compute_roc(train_visual_words, test_visual_words, train_labels, svm_train_y, classifier, RESULTS+'ROC_bow.png')

    accuracy = classifier.score(test_visual_words, val_labels)

    print(f'Test accuracy (n_clusters: {n_clusters}): {accuracy}')
    return accuracy


study = optuna.create_study(direction = "maximize")



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100) 
trial = study.best_trial
print("Best Score: ", trial.value)
print("Best Params: ")
for key, value in trial.params.items():
    print("  {}: {}".format(key, value))