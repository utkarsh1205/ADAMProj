import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import random
import _pickle as pickle
from dltk.io.augmentation import add_gaussian_noise, flip, extract_class_balanced_example_array
from dltk.io.preprocessing import whitening
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import FalsePositives, FalseNegatives, Precision, Recall, TruePositives, TrueNegatives
from tensorflow.keras.optimizers import Adam
from PIL import Image
# from resunet import ResUNet
from resunetpp import ResUnetPlusPlus
from metrics import Semantic_loss_functions
# from unet_3d import unet3d
from utility import *
import tensorflow.keras.backend as K

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

def dice(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score
    




test_directory = os.path.join('dataset1','test')

shape = (1,32,256,256)
s = Semantic_loss_functions()

model = tf.keras.models.load_model(os.path.join('best_model.hdf5'),custom_objects={'focal_tversky': s.focal_tversky, 'generalized_dice_coefficient':s.generalized_dice_coefficient})

pred_mask = predict_mask(sample='10059B', test_directory =test_directory, shape=shape, model=model)

image,mask = load_test_sample(os.path.join(test_directory,'10059B'))

score = dice(mask,pred_mask)

print('Dice Score:',score)

print('saving')

for j in range(image.shape[1]):
    fname = str(j)+'.jpeg'
    path = os.path.join('check_output','image',fname)
    image1 = (image - image.min())/(image.max()-image.min())
    image1 = image1*256
    im = Image.fromarray(image1[0,j,:,:])
    im = im.convert("L")
    im.save(path)
    
    
mask=mask[0]
for j in range(mask.shape[0]):
    fname = str(j)+'.jpeg'
    path = os.path.join('check_output','mask',fname)
    mask1 = mask*256
    im = Image.fromarray(mask1[j,:,:])
    im = im.convert("L")
    im.save(path)
    
for j in range(pred_mask.shape[0]):
    fname = str(j)+'.jpeg'
    path = os.path.join('check_output','pred',fname)
    mask1 = pred_mask*256
    im = Image.fromarray(mask1[j,:,:])
    im = im.convert("L")
    im.save(path)