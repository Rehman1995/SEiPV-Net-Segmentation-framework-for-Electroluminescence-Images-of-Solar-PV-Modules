# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:47:04 2023

@author: Rehman
"""

from main import mainmodel
from metricss import iou,dice_coef,jacard,precision_m,recall_m,f1_m,specificity,mean_iou
from loss_functions import multiclass_weighted_dice_loss, multiclass_weighted_tanimoto_loss, multiclass_weighted_squared_dice_loss,categorical_focal_loss

#%%


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%%

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#%%

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

#%%

#%% data prep


  
image_h = 512
image_w = 512
num_classes = 24
input_shape = (image_h, image_w, 3)
batch_size = 8# * strategy.num_replicas_in_sync
lr = 0.001 ## 0.0001
num_epochs = 30



global classes
global rgb_codes

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "train", "image", "*.png")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "valid", "image", "*.png")))
    valid_y = sorted(glob(os.path.join(path, "valid", "mask", "*.png")))

    test_x = sorted(glob(os.path.join(path, "test", "image", "*.png")))
    test_y = sorted(glob(os.path.join(path, "test", "mask", "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image_mask(x, y):
    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    
    x = x/255.0
    x = x.astype(np.float32)

    """ Mask """
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    y = cv2.resize(y, (image_w, image_h))
    
    y = y.astype(np.int32)

    return x, y

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()
        return read_image_mask(x, y)

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, num_classes)

    image.set_shape([image_h, image_w, 3])
    mask.set_shape([image_h, image_w, num_classes])

    return image, mask

def tf_dataset(X, Y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=5000).map(preprocess)
    ds = ds.batch(batch).prefetch(4)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files1")



    """ Paths """
    dataset_path ="/home/hasan/pvELI/data_24c/"
    model_path = os.path.join("files1", "model1.h5")
    csv_path = os.path.join("files1", "data1.csv")

    """ RGB Code and Classes """
    rgb_codes = [[0, 0, 0], [128,128,128], [80,80,80], [0, 0,255],[0,255,0], [100,50,50], [255, 0,100], [128,128,0],[255,215,0], [50,50,255], [0,255,255], [255,0,0], [255,0,255],  [255,255,0], [255,255,255],[255,165,0], [75,0,130], [32,32,32], [0,150,0], [218,165,32], [184,134,11], [127,255,215],[45,45,255], [50,50,50] ]

    classes = ["bckgnd","sp multi","sp mono","sp dogbone", "ribbons","border" , "text", "padding","clamp","busbars" , "crack rbn edge", "inactive", 	"rings", 	"material", 	"crack", 	"gridline","splice" ,	"dead cell" , 	"corrosion ","belt mark","edge dark", 	"frame edge", "jbox", 	"meas artifact"]

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)
    test_ds = tf_dataset(test_x, test_y, batch=batch_size)
    
    
    
    

#%% training
# use capital A in ablation study
#

#with strategy.scope():
model=mainmodel(24, input_height=image_h, input_width=image_w)
model.summary()

#model=A6(24, input_height=image_h, input_width=image_w)
#model.summary()

#input_shape = (512, 512, 3)
#model = unetm(input_shape,24)
#model.summary()

#model = PSPNet(inputs=(512,512,3), classes=24)
#model.summary()


#shape=(512,512,3)
#model=DeepLabv3plusX(shape,24)
#model.summary()
    


#model.compile(
        #loss=[categorical_focal_loss(alpha=[[[5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,
                                            # 5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,
                                              # 5.5,5.5,5.5,5.5]]], gamma=2)],#loss="categorical_crossentropy", #categorical_crossentropy
        
model.compile(
        loss=[ multiclass_weighted_squared_dice_loss( [0.15,0.25,0.25,0.25,0.30, 0.25,0.25,0.25,0.25,0.25,
                                             0.25,0.40,0.25,0.25,0.45, 0.35,0.25,0.25,0.25,0.25,
                                               0.25,0.25,0.25,0.25])],#loss="categorical_crossentropy", #categorical_crossentropy
                
        
        
         optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[dice_coef, iou,jacard, 
                              precision_m,
                            
                              recall_m,f1_m,specificity,
                              mean_iou ]
    )

callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0000001, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=False)
    ]

model.fit(train_ds,
        validation_data=valid_ds,
        epochs=num_epochs,
        callbacks=callbacks
    )



#%%
#%%

model.evaluate(test_ds)

#%% visual performance check


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
#sns.set()

#from train import create_dir, load_dataset



def grayscale_to_rgb(mask, rgb_codes):
    h, w = mask.shape[0], mask.shape[1]
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(rgb_codes[pixel])

    output = np.reshape(output, (h, w, 3))
    return output


def givemask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = grayscale_to_rgb(mask, rgb_codes)
    #mask= cv2.resize(mask, (512, 512))
    #mask =cv2.cvtColor(mask , cv2.COLOR_BGR2RGB)
    return mask

def givepred(pred):
    pred = np.expand_dims(pred , axis=-1)
    pred = grayscale_to_rgb(pred , rgb_codes)
    #pred= cv2.resize(pred, (512, 512))
    #pred =cv2.cvtColor(pred , cv2.COLOR_BGR2RGB)
    return pred 



def save_results(image_x, mask, pred, save_image_path):
    #mask = np.expand_dims(mask, axis=-1)
    #mask = grayscale_to_rgb(mask, rgb_codes)

    #pred = np.expand_dims(pred, axis=-1)
    #pred = grayscale_to_rgb(pred, rgb_codes)

    line = np.ones((image_x.shape[0], 10, 3)) * 255

    cat_images = np.concatenate([image_x, line, mask, line, pred], axis=1)
    #cat_images.save(save_image_path/cat_images)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results1")

    """ Hyperparameters """
    image_h = 512
    image_w =512
    num_classes = 24

    """ Paths """
    dataset_path = "/home/hasan/pvELI/data_24c/"   
    model_path = os.path.join("files1", "model1.h5")

    """ RGB Code and Classes """
    rgb_codes = [[0, 0, 0], [128,128,128], [80,80,80], [0, 0,255],[0,255,0], [100,50,50], [255, 0,100], [128,128,0],[255,215,0], [50,50,255], [0,255,255], [255,0,0], [255,0,255],  [255,255,0], [255,255,255],[255,165,0], [75,0,130], [32,32,32], [0,150,0], [218,165,32], [184,134,11], [127,255,215],[45,45,255], [50,50,50] ]

    classes = ["bckgnd","sp multi","sp mono","sp dogbone", "ribbons","border" , "text", "padding","clamp","busbars" , "crack rbn edge", "inactive", 	"rings", 	"material", 	"crack", 	"gridline","splice" ,	"dead cell" , 	"corrosion ","belt mark","edge dark", 	"frame edge", "jbox", 	"meas artifact"]

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'iou':iou,'jacard':jacard,'multiclass_weighted_squared_dice_loss':multiclass_weighted_squared_dice_loss ,#'categorical_focal_loss': categorical_focal_loss,
                            
                              'precision_m':precision_m,'recall_m':recall_m,'f1_m':f1_m,'specificity':specificity,'mean_iou':mean_iou}):
        
        model = tf.keras.models.load_model(model_path)

    """ Prediction & Evaluation """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image_w, image_h))
        image_x = image
        image = image/255.0 ## (H, W, 3)
        image = np.expand_dims(image, axis=0) ## [1, H, W, 3]
        image = image.astype(np.float32)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_w, image_h))
        mask = mask.astype(np.int32)
        #mask = grayscale_to_rgb(mask, rgb_codes)

        """ Prediction """
        pred = model.predict(image, verbose=0)[0]
        pred = np.argmax(pred, axis=-1) ## [0.1, 0.2, 0.1, 0.6] -> 3
        pred = pred.astype(np.int32)
        #pred = grayscale_to_rgb(pred, rgb_codes)

        ## cv2.imwrite("pred.png", pred * (255/11))


        # rgb_mask = grayscale_to_rgb(pred, rgb_codes)
        # cv2.imwrite("pred.png", rgb_mask)
        mask=givemask(mask)
        pred=givepred(pred)
        """ Save the results """
        save_image_path = f"results1/{name}.png"
        save_results(image_x, mask, pred, save_image_path)


        fig, ax = plt.subplots(1, 3, figsize = (15, 15))

        #plt.imshow(pred)

        ax[0].imshow(mask , cmap = 'jet' )
        ax[0].set_title(f'GT ', fontsize = 15)
        ax[0].axis("off")

        ax[1].imshow(pred , cmap = 'jet' )
        ax[1].set_title('Prediction', fontsize = 15)
        ax[1].axis("off")

        ax[2].imshow(image_x , cmap = 'gray')
        ax[2].set_title(f'original ', fontsize = 15)
        ax[2].axis("off")

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
