# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:41:10 2023

@author: jbl
"""


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from typing import Callable, Union
import numpy as np

#%%

def multiclass_weighted_dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
   
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def multiclass_weighted_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
   
        axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights # Broadcasting
        denominator = K.sum(denominator, axis=axis_to_reduce)

        return 1 - numerator / denominator

    return multiclass_weighted_dice_loss

#%%

def multiclass_weighted_tanimoto_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def multiclass_weighted_tanimoto_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
       
        axis_to_reduce = range(1, K.ndim(y_pred))  # All axis but first (batch)
        numerator = y_true * y_pred * class_weights
        numerator = K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true**2 + y_pred**2 - y_true * y_pred) * class_weights
        denominator = K.sum(denominator, axis=axis_to_reduce)
        return 1 - numerator / denominator

    return multiclass_weighted_tanimoto_loss

#%%



def multiclass_weighted_squared_dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor],
                                                                                                   tf.Tensor]:
  
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def multiclass_weighted_squared_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
      
        axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true**2 + y_pred**2) * class_weights  # Broadcasting
        denominator = K.sum(denominator, axis=axis_to_reduce)

        return 1 - numerator / denominator

    return multiclass_weighted_squared_dice_loss

#%%





import numpy as np
from keras import backend as K
import tensorflow as tf

#import dill




def categorical_focal_loss(alpha, gamma=2.):
   
 

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss(y_true, y_pred):
        
  

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss
