import numpy as np
import os
import sys
from keras.models import Model
from keras.layers import *
from keras.engine import Layer
from keras.models import *
import keras.backend as K

#data variables
data_dir = 'data'

orig_train = np.load(os.path.join(data_dir,'ct_train.npy'))
orig_patches_train = np.load(os.path.join(data_dir,'y_patches_train.npy'))

test_val, y_patches_val = orig_train[33822:], orig_patches_train[33822:]
ct_train, y_patches_train = orig_train[:33822], orig_patches_train[:33822]


#hyperparameters
hm_epochs = 5
batch_size = 18

#initializing model
from model import UNET_2D
model = UNET_2D(ct_train)

#loss/evaluation
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#compiling model
import keras
callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0003, patience=2, verbose=2, mode='auto')
adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = adam, loss = dice_coef_loss, metrics = [dice_coef])

#training with data augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip=True)
datagen.fit(ct_train,augment = True)
history = model.fit_generator(datagen.flow(ct_train, y_patches_train, batch_size=batch_size),
                            steps_per_epoch=len(ct_train) / batch_size, epochs=hm_epochs, callbacks= [callback], validation_data 
                            datagen.flow(val_train, y_patches_val, batch_size=batch_size), validation_steps = len(val_train)/4)
#saving model
model.save('model.h5')
