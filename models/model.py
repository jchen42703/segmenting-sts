import numpy as np
import os
import sys
from keras.models import Model
from keras.layers import *
from keras.engine import Layer
from keras.models import *
import keras.backend as K

#defining models architecture
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

class MultichannelCascadeCNN(object):
    """
    from : https://github.com/dalmia/Brain-Tumor-Segmentation-Keras
    
    A Multi-channel cascaded CNN architecture introduced in 
    https://arxiv.org/pdf/1505.03540.pdf - "Brain Tumor Segmentation with Deep Neural Networks".
    
    PARAMETERS
    ----------
    patch_size: int, default=33
                Size of the patch fed into the model.
    
    mode: {'input', 'local', 'final'}, default='input'
          The type of cascaded to be done (Refer the README for clarity).
          
    num_channels: int, default=1
                  Number of channels in the input patch.
    
    num_classes: int, default=2
                 Number of possible classes for a pixel.
                 
    num_filters_local_1: int, default=64
                         Number of filters to be used in the first convolutional layer
                         of the local pathway.
                         
    num_filters_local_2: int, default=64
                         Number of filters to be used in the second convolutional layer
                         of the local pathway.
                         
    num_filters_global: int, default=160
                         Number of filters to be used in the convolutional layer of the
                         global pathway.
                         
    kernel_local_1: int, default=7
                    Kernel size to be used in the first convolutional layer of the local
                    pathway.
                    
    kernel_local_2: int, default=3
                    Kernel size to be used in the second convolutional layer of the local
                    pathway.
                
    kernel_global: int, default=13
                   Kernel size to be used in the convolutional layer of the global pathway.
    
    pool_local_1: int, default=4
                  The pooling size for the max pooling layer after the first convolutional 
                  layer of the local pathway.
    
    pool_local_2: int, default=2
                  The pooling size for the max pooling layer after the second convolutional 
                  layer of the local pathway.
    
    
    """
    
    def __init__(self, patch_size=33, mode='input', num_channels=1, num_classes=2, 
                 num_filters_local_1=64, num_filters_local_2=64, num_filters_global=160, 
                 kernel_local_1=7, kernel_local_2=3, kernel_global=13, pool_local_1=4,
                 pool_local_2=2):
        
        self.patch_size = patch_size
        self.mode = mode
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_filters_local_1 = num_filters_local_1
        self.num_filters_local_2 = num_filters_local_2
        self.num_filters_global = num_filters_global
        self.kernel_local_1 = kernel_local_1
        self.kernel_local_2 = kernel_local_2
        self.kernel_global = kernel_global
        self.pool_local_1 = pool_local_1
        self.pool_local_2 = pool_local_2
        
        self.classification_kernel_size = self.patch_size - self.kernel_global + 1
        
        if(self.mode == 'input'):
            self.outer_patch_size = 2 * self.patch_size - 1
            
        elif(self.mode == 'local'):
            self.outer_patch_size = self.patch_size + self.classification_kernel_size + \
                self.pool_local_2 + self.kernel_local_2 - 3
        
        else:
            self.outer_patch_size = self.patch_size + self.classification_kernel_size - 1
            
            
    def _forward_pass(self, model_input, stage, append_output=None, prev_output=None):
        
        local_kernel_1 = (self.kernel_local_1, self.kernel_local_1)
        local_kernel_2 = (self.kernel_local_2, self.kernel_local_2)
        global_kernel = (self.kernel_global, self.kernel_global)

        local_output_dim_1 = self.num_filters_local_1
        local_output_dim_2 = self.num_filters_local_2
        global_output_dim = self.num_filters_global

        local_pool_1 = (self.pool_local_1, self.pool_local_1)
        local_pool_2 = (self.pool_local_2, self.pool_local_2)
        
        # InputCascadeCNN
        if(append_output == 'input'): 
            final_input = Concatenate(axis=-1, name='input_concat')([model_input, prev_output])

        else:
            final_input = model_input

        local_output = MaxoutConv2D(local_kernel_1, output_dim=local_output_dim_1, 
                                    name=stage+'_local_conv1')(final_input)

        local_output = MaxPooling2D(pool_size=local_pool_1, strides=(1, 1),
                                    name=stage+'_local_pool1')(local_output)
        
        # LocalCascadeCNN
        if(append_output == 'local'):
            local_output = Concatenate(axis=-1, name='local_concat')([local_output, prev_output])

        local_output = MaxoutConv2D(local_kernel_2, output_dim=local_output_dim_2, 
                                    name=stage+'_local_conv2')(local_output)
        local_output = MaxPooling2D(pool_size=local_pool_2, strides=(1, 1), 
                                    name=stage+'_local_pool2')(local_output)

        global_output = MaxoutConv2D(global_kernel, output_dim=global_output_dim, 
                                     name=stage+'_global')(final_input)

        output = Concatenate(axis=-1, name=stage+'_concat')([local_output, global_output])      
        
    def build_model(self):
        """
        Builds the MultiCascadedCNN model.
        
        Returns
        -------
        
        model: Keras Model object
               The MultiCascadedCNN model as per the mode.
               
        """
        
        # First stage
        first_cascade_input_shape = (self.outer_patch_size, 
                                     self.outer_patch_size, 
                                     self.num_channels)
    
        first_cascade_input = Input(shape=first_cascade_input_shape)
        first_cascade_output = self._forward_pass(first_cascade_input, stage='1')    
        
        # Second stage
        second_cascade_input_shape = (self.patch_size, self.patch_size, self.num_channels)
        second_cascade_input = Input(shape=second_cascade_input_shape)
        second_cascade_output = self._forward_pass(second_cascade_input, 
                                             stage='2',
                                             append_output=self.mode,
                                             prev_output=first_cascade_output)

        output = Flatten()(second_cascade_output)
        model = Model(inputs=[first_cascade_input, second_cascade_input], outputs=output)
        
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
