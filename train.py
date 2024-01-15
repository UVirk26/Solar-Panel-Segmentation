import numpy as np 
import pandas as pd 
import shutil
import tensorflow as tf
from zipfile import ZipFile
import keras.backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow import keras
from keras import layers
from utils import *
from model import *
from preprocessing import *

model = unet_model(1)

model.compile(optimizer='adam',
              loss = dice_loss,
              metrics=[dice_coef,'binary_accuracy'])

print(model.summary())

model.save_weights("model.h5")
print("Saved model to disk")
