import numpy as np
import os
import sys
from keras.models import Model
from keras.layers import *
from keras.engine import Layer
from keras.models import *
import keras.backend as K

#credits to dalmia for the code for this layer
class MaxoutConv2D(Layer):
    """
    Convolution Layer followed by Maxout activation as described 
    in https://arxiv.org/abs/1505.03540.
    
    PARAMETERS
    ----------
    
    kernel_size: kernel_size parameter for Conv2D
    output_dim: final number of filters after Maxout
    keep_prob: keep probability for Dropout
    nb_features: number of filter maps to take the Maxout over; default=4
    padding: 'same' or 'valid'
    first_layer: True if x is the input_tensor
    input_shape: Required if first_layer=True
    
    """
    
    def __init__(self, kernel_size, output_dim, nb_features=4, padding='valid', **kwargs):
        
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.nb_features = nb_features
        self.padding = padding
        super(MaxoutConv2D, self).__init__(**kwargs)

    def call(self, x):

        num_channels = self.output_dim * self.nb_features
        conv_out = Conv2D(num_channels, self.kernel_size, padding=self.padding)(x)
        batch_norm_out = BatchNormalization()(conv_out)
        out_shape = batch_norm_out.get_shape().as_list()
        reshape_out = Reshape((out_shape[1], out_shape[2], 
                               self.nb_features, self.output_dim))(batch_norm_out)
        maxout_out = K.max(reshape_out, axis=-2)

        return maxout_out

    def get_config(self):

        config = {"kernel_size": self.kernel_size,
                  "output_dim": self.output_dim,
                  "nb_features": self.nb_features,
                  "padding": self.padding}

        base_config = super(MaxoutConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        input_height= input_shape[1]
        input_width = input_shape[2]
        
        if(self.padding == 'same'):
            output_height = input_height
            output_width = input_width
        
        else:
            output_height = input_height - self.kernel_size[0] + 1
            output_width = input_width - self.kernel_size[1] + 1
        
        return (input_shape[0], output_height, output_width, self.output_dim)

#defining model architectures

def UNET_2D(x_train): 
    '''
    2d UNet implementation
    '''
    inputs = Input(x_train.shape[1:])
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    conv9 = Conv2D(32, (1, 1), activation='relu')(conv8)
    conv10 = Conv2D(1, (1,1), activation = 'sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def cascade(first_cascade, second_cascade):
    ''' 
    Arguments:
        first_cascade: input array for the first cascade (56 by 56 patches)
        second_cascade: input array for the second cascade (28 by 28 patches)
        NOTE: TO GET THE SECOND_CASCADE PATCHES, JUST CHANGE THE PATCH DIMENSIONS IN preprocess.py and run data-gen.py
    Returns:
        model: keras class 
    '''
    
    #first cascade
    first_cascade_input = Input(first_cascade.shape[1:])
    conv_1 = MaxoutConv2D(kernel_size = (7,7), output_dim=64, name = 'conv_1')(first_cascade_input)
    maxpool_1 = MaxPooling2D(pool_size=4, strides=(1, 1), 
                                     name='local_pool1')(conv_1)
    
    conv_2 = MaxoutConv2D(kernel_size = (3,3), output_dim=64, name = 'conv_2')(maxpool_1)
    maxpool_2 = MaxPooling2D(pool_size=2, strides=(1, 1), 
                                     name='local_pool2')(conv_2)
   
    global_output = MaxoutConv2D((13,13), output_dim=160, padding = 'same',
                                     name='global')(maxpool_2)
    
    output = Concatenate(axis=-1, name='concat')([maxpool_2, global_output])
    output = Conv2D(2, (17,17), padding='valid', name='conv_last')(output)
    output = Activation('softmax', name='softmax')(output)   
    
    #second cascade (input mode)
    second_cascade_input = Input(second_cascade.shape[1:])
    final_input = Concatenate(axis=-1, name='input_concat')([second_cascade_input, output])
    conv2_1 = MaxoutConv2D(kernel_size = (7,7), output_dim=64, name = '2_conv_1')(final_input)
    maxpool2_1 = MaxPooling2D(pool_size=4, strides=(1, 1), 
                                     name='2_local_pool1')(conv2_1)
    
    conv2_2 = MaxoutConv2D(kernel_size = (3,3), output_dim=64, name = '2_conv_2')(maxpool2_1)
    maxpool2_2 = MaxPooling2D(pool_size=2, strides=(1, 1), 
                                     name='2_local_pool2')(conv2_2)
   
    global2_output = MaxoutConv2D((13,13), output_dim=160, padding = 'same',
                                     name='2_global')(maxpool2_2)
    
    output2 = Concatenate(axis=-1, name='2_concat')([maxpool2_2, global2_output])
    output2 = Conv2D(2, (16,16), padding='valid', name='2_conv_last')(output2)
    output2 = Activation('softmax', name='2_softmax')(output2)   
    
    final_output = Flatten()(output2)
    model = Model(inputs=[first_cascade_input, second_cascade_input], outputs=final_output)
    
    return model

def dilated(x_train):
    '''
    deprecated syntax
    '''
    model_in = Input(x_train.shape[1:])
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, name='final')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
    h = ZeroPadding2D(padding=(4, 4))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
    h = ZeroPadding2D(padding=(8, 8))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
    h = ZeroPadding2D(padding=(16, 16))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
    h = ZeroPadding2D(padding=(32, 32))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(32, 32), activation='relu', name='ctx_conv6_1')(h)
    h = ZeroPadding2D(padding=(64, 64))(h)
    h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(64, 64), activation='relu', name='ctx_conv7_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_fc1')(h)
    h = Convolution2D(classes, 1, 1, name='ctx_final')(h)

    # the following two layers pretend to be a Deconvolution with grouping layer.
    # never managed to implement it in Keras
    # since it's just a gaussian upsampling trainable=False is recommended
    h = UpSampling2D(size=(8, 8))(h)
    logits = Convolution2D(classes, 16, 16, border_mode='same', bias=False, trainable=False, name='ctx_upsample')(h)

    if apply_softmax:
        model_out = softmax(logits)
    else:
        model_out = logits

    model = Model(input=model_in, output=model_out)

    return model
