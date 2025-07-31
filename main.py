# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:44:17 2023 

@author: Rehman 
""" 



#%% main model

import tensorflow as tf 
from keras.models import *
from keras.layers import *

def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(2,2), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(4,4), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    #out= Dropout(0.3)(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    #out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    #out = BatchNormalization()(out)
    #out = Activation('relu')(out)
   # out= Dropout(0.3)(out)
    return out

def Attention_block(input1, input2, filters):
    g1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input1)
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(filters, kernel_size=1, strides=1, padding='same')(input2)
    x1 = BatchNormalization()(x1)
    psi = Activation('relu')(add([g1, x1]))
    psi = Conv2D(filters, kernel_size=1, strides=1, padding='same')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)
    #psi= Dropout(0.5)(psi)
    out = multiply([input2, psi])
    return out
    

def de2(input, filters):
 


    out31 = Conv2D(filters, kernel_size=(2,2), strides=1, padding='same')(input)
    out31 = BatchNormalization()(out31)
    out31 = Activation('relu')(out31)

    out32 = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out32 = BatchNormalization()(out32)
    out32 = Activation('relu')(out32)

    ycc1 = Concatenate()([out31,out32])

    out33 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(ycc1)
    out33 = BatchNormalization()(out33)
    out33 = Activation('relu')(out33)


    out34 = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out34 = BatchNormalization()(out34)
    out34 = Activation('relu')(out34)
    out34 = Conv2D(filters, kernel_size=(2,2), strides=1, padding='same')(out34)
    out34 = BatchNormalization()(out34)
    out34= Activation('relu')(out34)

    ycc2 = Concatenate()([out33,out34])

    out35 = Conv2D(filters, kernel_size=(2,2), strides=1, padding='same')(ycc2)
    out35 = BatchNormalization()(out35)
    out35 = Activation('relu')(out35)
    #out= Dropout(0.3)(out)
    return out35

def ce1(input, filters):
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(2,2), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(4,4), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

   # out1 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    #out1 = BatchNormalization()(out1)
    #out1 = Activation('relu')(out1)


    #x3 = concatenate([out, out1])
    #out= Dropout(0.3)(out)
    return out


def conv_blockU1(input, filters):
    

    out12 = Conv2D(filters, kernel_size=(9,9), strides=1, padding='same')(input)
    out12 = BatchNormalization()(out12)
    out12 = Activation('relu')(out12)
    out12 = Conv2D(filters, kernel_size=(7,7), strides=1, padding='same')(out12)
    out12 = BatchNormalization()(out12)
    out12 = Activation('relu')(out12)

    out21 = Conv2D(filters, kernel_size=(5,5), strides=1, padding='same')(out12)
    out21 = BatchNormalization()(out21)
    out21 = Activation('relu')(out21)
    out21 = Conv2D(filters, kernel_size=(4,4), strides=1, padding='same')(out21)
    out21 = BatchNormalization()(out21)
    out21 = Activation('relu')(out21)

    out31 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out21)
    out31 = BatchNormalization()(out31)
    out31 = Activation('relu')(out31)
    out31 = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(out31)
    out31 = BatchNormalization()(out31)
    out31 = Activation('relu')(out31)
    
    shape = input.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(input)
    y_pool = Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    out311 = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out311 = BatchNormalization()(out311)
    out311 = Activation('relu')(out311)

    o1 = Concatenate()([out31,y_pool,out311])


    return o1

def cek1(input, filters):
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(4,4), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    

   # out1 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    #out1 = BatchNormalization()(out1)
    #out1 = Activation('relu')(out1)


    #x3 = concatenate([out, out1])
    #out= Dropout(0.3)(out)
    return out



def de311(input, filters, p1=2,p3=4):


    shape = input.shape
    

    y_61 = Conv2D(filters, kernel_size=1, padding='same', dilation_rate = 3,use_bias=False)(input)#6
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
 
    y_62 = Conv2D(filters, kernel_size=1, padding='same', dilation_rate = 1,use_bias=False)(y_61)#1
    y_62 = BatchNormalization()(y_62)
    y_62 = Activation('relu')(y_62)

    y_622 = Conv2D(filters, kernel_size=1, padding='same', dilation_rate = 6,use_bias=False)(input)#1
    y_622 = BatchNormalization()(y_622)
    y_622 = Activation('relu')(y_622)

    red = GlobalAveragePooling2D()(input)
    red = Reshape((1,1,shape[-1]))(red)
    red = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False)(red)
    red = BatchNormalization()(red)
    red = Activation('relu')(red)
    red = UpSampling2D(size=shape[1],interpolation='bilinear')(red)

    orange = AveragePooling2D(pool_size=(p1))(input)
    orange = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False)(orange)
    orange = BatchNormalization()(orange)
    orange = Activation('relu')(orange)
    orange = UpSampling2D(size=p1,interpolation='bilinear')(orange)

    green = AveragePooling2D(pool_size=(p3))(input)
    green = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False)(green)
    green = BatchNormalization()(green)
    green = Activation('relu')(green)
    green = UpSampling2D(size=p3,interpolation='bilinear')(green)

    y_cc3 = Concatenate()([red, green , orange , y_62,y_622])
    

    #out= Dropout(0.3)(out)
    return y_cc3

def cek(input, filters):
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(filters, kernel_size=(7,7), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(7,7), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(filters, kernel_size=(5,5), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(5,5), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    
    

   # out1 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    #out1 = BatchNormalization()(out1)
    #out1 = Activation('relu')(out1)


    #x3 = concatenate([out, out1])
    #out= Dropout(0.3)(out)
    return out


from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, Concatenate, Input, Reshape
from keras.models import Model

def de31(input, filters, p1=2,p3=4,p31=6):


    shape = input.shape
    

    y_61 = Conv2D(filters, kernel_size=1, padding='same', dilation_rate = 3,use_bias=False)(input)#6
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
 
    y_62 = Conv2D(filters, kernel_size=1, padding='same', dilation_rate = 1,use_bias=False)(y_61)#1
    y_62 = BatchNormalization()(y_62)
    y_62 = Activation('relu')(y_62)
    
    y_621 = Conv2D(filters, kernel_size=1, padding='same', dilation_rate = 9,use_bias=False)(input)#1
    y_621 = BatchNormalization()(y_621)
    y_621 = Activation('relu')(y_621)

    red = GlobalAveragePooling2D()(input)
    red = Reshape((1,1,shape[-1]))(red)
    red = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False)(red)
    red = BatchNormalization()(red)
    red = Activation('relu')(red)
    red = UpSampling2D(size=shape[1],interpolation='bilinear')(red)

    orange = AveragePooling2D(pool_size=(p1))(input)
    orange = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False)(orange)
    orange = BatchNormalization()(orange)
    orange = Activation('relu')(orange)
    orange = UpSampling2D(size=p1,interpolation='bilinear')(orange)

    green = AveragePooling2D(pool_size=(p3))(input)
    green = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False)(green)
    green = BatchNormalization()(green)
    green = Activation('relu')(green)
    green = UpSampling2D(size=p3,interpolation='bilinear')(green)
    
    green1 = AveragePooling2D(pool_size=(p31))(input)
    green1 = Conv2D(filters, kernel_size=(1,1), padding='same', use_bias=False)(green1)
    green1 = BatchNormalization()(green1)
    green1 = Activation('relu')(green1)
    green1 = UpSampling2D(size=p31,interpolation='bilinear')(green1)

    y_cc3 = Concatenate()([red, green , orange , y_62,y_621,green1])# green1, y_621
    

    #out= Dropout(0.3)(out)
    return y_cc3

def de3(input, filters):


    shape = input.shape
    
    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(input)
    y_pool = Conv2D(filters, kernel_size=3, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_6 = Conv2D(filters, kernel_size=3, padding='same', dilation_rate = 6,use_bias=False)(input)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)
    
    
    y_61 = Conv2D(filters, kernel_size=3, padding='same', dilation_rate = 9,use_bias=False)(input)#6
    y_61 = BatchNormalization()(y_61)
    y_61 = Activation('relu')(y_61)
 
    y_62 = Conv2D(filters, kernel_size=3, padding='same', dilation_rate = 12,use_bias=False)(input)#1
    y_62 = BatchNormalization()(y_6)
    y_62 = Activation('relu')(y_6)


    y_cc3 = Concatenate()([y_pool, y_6 , y_61 , y_62])
    

    #out= Dropout(0.3)(out)
    return y_cc3

def ce11(input, filters):
    out = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out1 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)

    ca= Concatenate()([input, out1])

    out2 = Conv2D(filters, kernel_size=(2,2), strides=1, padding='same')(ca)# 4,4
    out2 = BatchNormalization()(out2)
    out2 = Activation('relu')(out2)

    out3 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out2)#3,3
    out3 = BatchNormalization()(out3)
    out3 = Activation('relu')(out3)

    cb= Concatenate()([input, out1,out3])

    out31 = Conv2D(filters, kernel_size=(1,1), strides=1, padding='same')(cb)
    out31 = BatchNormalization()(out31)
    out31 = Activation('relu')(out31)

    #out= Dropout(0.3)(out)
    return out31

def ce111(input, filters):
    out = Conv2D(filters, kernel_size=(5,5), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out1 = Conv2D(filters, kernel_size=(7,7), strides=1, padding='same')(out)
    out1 = BatchNormalization()(out1)
    out1 = Activation('relu')(out1)

    ca= Concatenate()([input, out1])

    out2 = Conv2D(filters, kernel_size=(2,2), strides=1, padding='same')(ca)#3,3
    out2 = BatchNormalization()(out2)
    out2 = Activation('relu')(out2)

    out3 = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out2)#1,1
    out3 = BatchNormalization()(out3)
    out3 = Activation('relu')(out3)

    #out= Dropout(0.3)(out)
    return out3


def mainmodel(nClasses, input_height=224, input_width=224):
    
    inputs = Input(shape=(input_height, input_width, 3))
    n1 = 6
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 12]
    #filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 4]

    e1 =ce111(inputs, filters[0])
    e2 = MaxPooling2D(strides=2)(e1)
    
    e2 =  ce111(e2, filters[1])
    e3 = MaxPooling2D(strides=2)(e2)
    
    e3 =  ce111(e3, filters[2])
    e4 = MaxPooling2D(strides=2)(e3)
    
    e4 =  ce111(e4, filters[3])
    e5 = MaxPooling2D(strides=2)(e4)
    
    e5 = ce111(e5, filters[4]) # ce111

    e11=ce1(e1, filters[0])#ce1
    e22=ce1(e2, filters[1])
    e33=ce1(e3, filters[2])
    e44=ce1(e4, filters[3])

    #e55=de3(e5, filters)

    #e111 = Concatenate()([e11, e1])
    #e222 = Concatenate()([e22, e2])
    #e333 = Concatenate()([e33, e3])
    #e444 = Concatenate()([e44, e4])


    d5 = up_conv(e5, filters[3])
    x4 =  Attention_block(d5, e44, filters[3])
    d5 = Concatenate()([x4, d5])
    #d5 = conv_block(d5, filters[3])
    d5 =de311(d5, filters[3])

    d4 = up_conv(d5, filters[2])
    x3 =  Attention_block(d4, e33, filters[2])
    d4 = Concatenate()([x3, d4])
    #d4 = conv_block(d4, filters[2])
    d4 =de311(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    x2 =  Attention_block(d3, e22, filters[1])
    d3 = Concatenate()([x2, d3])
    #d3 = conv_block(d3, filters[1])
    d3 =de311(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    x1 =  Attention_block(d2, e11, filters[0])
    d2 = Concatenate()([x1, d2])
    #d2 = conv_block(d2, filters[0])
    d2 =de311(d2, filters[0])

    #d2= Dropout(0.2)(d2)

    o = Conv2D(nClasses, (3,3), padding='same')(d2)#, 3*3 filtersize

    out = Activation('softmax')(o)

    model = Model(inputs, out)
    

    return model

#%%
#model =mainmodel(24, input_height=224, input_width=224)
#model.summary()

#%%















