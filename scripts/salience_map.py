import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix

import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings

import argparse
from keras.models import load_model

#parser = default_argument_parser()
parser = argparse.ArgumentParser()
parser.add_argument("-weights-path", type=str, default = "/content/drive/MyDrive/effnet_tumor_class.h5", metavar="FILE", help="trained model with weights")
parser.add_argument("-input-image-path", type=str, default ="/content/drive/MyDrive/tumor_dataset/Testing/no_tumor/image(1).jpg", metavar="FILE", help="sample image to visualise the salience map")

args = parser.parse_args()
print("Command Line Args:", args)

weights_path = args.weights_path
model = load_model(weights_path)
print("model_fetched is : ",model)

img_path = args.input_image_path
image_size =150


successive_outputs = [layer.output for layer in model.layers[1:]]

visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# this is a PIL image
img = tf.keras.utils.load_img(img_path, target_size=(image_size, image_size)) 

# Numpy array with shape (150, 150, 3)
x   = tf.keras.utils.img_to_array(img)                         

# Numpy array with shape (1, 150, 150, 3)
x   = x.reshape((1,) + x.shape)                

#x /= 255.0
#getting the feature maps from the model
successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  #print(feature_map.shape)
  if len(feature_map.shape) == 4:
    
    # Applicable for the conv / maxpool layers and not for fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # tiling the images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    np.seterr(invalid='ignore')

    #post processing the features to make it visually compatible
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #display grid
    scale = 20. / n_features
    #plt.plot(range(10))
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
    plt.show()