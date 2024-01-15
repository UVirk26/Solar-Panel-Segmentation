import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras

import os
import cv2
import numpy as np

from PIL import Image
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from keras import layers

from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
import tensorflow as tf

import pandas as pd 
import shutil
from zipfile import ZipFile
import keras.backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

root_dir = '/kaggle/input/solar-panel-detection-and-identification/PV03'
resolution = 'PV03'
data_dir = os.path.join(root_dir)#,resolution)

image_root = '/kaggle/working/image'
label_root = '/kaggle/working/masks'
valid_image = '/kaggle/working/val_image'
valid_mask = '/kaggle/working/val_masks'

if not os.path.isdir(image_root):
    os.mkdir(image_root)
if not os.path.isdir(label_root):
    os.mkdir(label_root)

images = list()
labels = list()

for (dirpath, dirnames, filenames) in os.walk(data_dir):
    # img_names += [os.path.join(dirpath, file) for file in filenames]
    images += [os.path.join(dirpath, file) for file in filenames]

labels += [i for i in filter(lambda score: '_label.bmp' in score, images)]
images = [i for i in filter(lambda score: '_label.bmp' not in score, images)]

i = 0
for img_path in images:
    i+=1
    src_path = img_path
    dst_path = os.path.join(image_root,os.path.basename(img_path))
    dst_path_val = os.path.join(valid_image,os.path.basename(img_path))
    img = cv2.imread(src_path)
    new_img = cv2.resize(img,(256, 256))
    if i <2200:
        cv2.imwrite(dst_path[:-4]+'.png',new_img)
    else:
        cv2.imwrite(dst_path_val[:-4]+'.png',new_img)


i = 0
for label_path in labels:
    i+=1
    src_path = label_path
    file_name = os.path.basename(label_path).replace('_label','')
    dst_path = os.path.join(label_root,file_name)
    dst_path_val = os.path.join(valid_mask,file_name)
    img = cv2.imread(src_path)
    new_img = cv2.resize(img,(256, 256))
    new_img[new_img>=100] = 1
    new_img[new_img!=1] = 0
    if i <2200:
        cv2.imwrite(dst_path[:-4]+'.png',new_img[:,:,0])
    else:
        cv2.imwrite(dst_path_val[:-4]+'.png',new_img[:,:,0])


import cv2
from glob import glob

DATA_DIR = '/kaggle/working/'
train_images = sorted(glob(os.path.join(DATA_DIR, "image/*")))
train_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))
val_images = sorted(glob(os.path.join(DATA_DIR, "val_image/*")))
val_masks = sorted(glob(os.path.join(DATA_DIR, "val_masks/*")))

IMAGE_SIZE = 256
BATCH_SIZE = 4
def read_image(image_path, mask=False):
    image = tf_io.read_file(image_path)
    if mask:
        image = tf_image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def data_generator(image_list, mask_list):
    dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

metric=['accuracy']

def get_model():
    return DeeplabV3Plus(256,2)

model = get_model()
# model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
    loss=loss,
    metrics=metric,
)

history1 = model.fit(train_dataset, validation_data=val_dataset, 
                    batch_size = 4, 
                    verbose=1,
                    epochs=40, 
                    shuffle=False)

model.save('satellite_standard_deeplabv3_100epochs_7May2021.hdf5')

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def predictions(image_list):
    x = []
    for image_file in image_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        x.append(prediction_mask)
    return x

y_pred=predictions(val_images)

    
