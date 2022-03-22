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

path = args.input_image_path
image_size =150

img = cv2.imread(path)
img = cv2.resize(img,(image_size,image_size))

prediction_image=np.array(img)
prediction_image= np.expand_dims(img, axis=0)
reverse_mapping={0:'pituitary_tumor', 1:'no_tumor', 2:'meningioma_tumor', 3:'glioma_tumor'}

def mapper(value):
    return reverse_mapping[value]

prediction=model.predict(prediction_image)
value=np.argmax(prediction)
move_name=mapper(value)
print("")
print("Prediction is: {}.".format(move_name))
