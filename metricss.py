# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:43:27 2023

@author: jbl
"""


import numpy as np
#import tensorflow as tf
from tensorflow.keras import backend as K
#import keras.backend as K
import sklearn

EPS = 1e-12

def get_iou(gt, pr, n_classes=29):
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

import tensorflow as tf
import numpy as np 
 
#if tf.__version__ == '1.13.1' or tf.__version__ == '1.15.0': 
#    from keras.layers.merge import concatenate
#    from keras.layers import Activation
#    import keras.backend as K 
#if tf.__version__ == '2.2.0' or tf.__version__ == '2.1.0' or tf.__version__ == '2.3.0' or tf.__version__ == '2.5.0': 
import keras.backend as K
from keras.layers import Activation, concatenate
    #import tensorflow_addons as tfa



from keras import backend as K
from keras import backend as keras

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 
    K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def jacard(y_true, y_pred):

  y_true_f = keras.flatten(y_true)
  y_pred_f = keras.flatten(y_pred)
  intersection = keras.sum ( y_true_f * y_pred_f)
  union = keras.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

  return intersection/union

from tensorflow.keras.metrics import binary_accuracy


def accuracy(y_true, y_pred):
    class_nums = y_pred.shape[-1]//2

    y_true = y_true[..., class_nums:]
    y_pred = y_pred[..., class_nums:]
    bi_acc = binary_accuracy(y_true, y_pred)

    return

import numpy as np
import tensorflow as tf
from keras import backend as K



def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

import numpy as np




def mean_iou(y_true, y_pred, smooth=1):
    
    
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou

def mean_ioucrack(y_true, y_pred, smooth=1):
    
    y_true = K.flatten(y_true[:,:,:,14])
    y_pred = K.flatten(y_pred[:,:,:,14])

    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou

def mean_iougrid(y_true, y_pred, smooth=1):
    
    y_true = K.flatten(y_true[:,:,:,15])
    y_pred = K.flatten(y_pred[:,:,:,15])

    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1,0])
    
    return iou

from tensorflow.keras import backend as K
from sklearn.metrics import jaccard_score,confusion_matrix

def jacard(y_true, y_pred):

    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum ( y_true_f * y_pred_f)
    union = keras.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union