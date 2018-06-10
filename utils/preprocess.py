import os
import random
import tensorflow as tf
import numpy as np
import h5py 

#parameters
PATCH_HEIGHT = 56 # actual: 28
PATCH_WIDTH = 56 # actual: 28
PATCH_SHAPE = [-1, PATCH_HEIGHT, PATCH_WIDTH, 1]
PATCH_SIZE = [1, PATCH_HEIGHT, PATCH_WIDTH, 1]
PATCH_STRIDES = [1, 1, 1, 1]
PATCH_RATES = [1, 1, 1, 1]
PATCH_PADDING = 'SAME'
SEED = 1

TRAIN_PATIENTS = ['STS_002','STS_005','STS_021','STS_023','STS_031',]
TEST_PATIENTS = ['STS_003','STS_012',]

patients = {
    'train': TRAIN_PATIENTS,
    'test': TEST_PATIENTS,
}

POSITIVE_SLICES = {
    'STS_002': (53, 63),
    'STS_003': (9, 24),
    'STS_005': (90, 123),
    'STS_012': (11, 39),
    'STS_021': (150, 189),
    'STS_023': (121, 172),
    'STS_031': (11, 41),
}

np.random.seed(SEED)

#loading data
data = h5py.File(os.path.join('input','lab_petct_vox_5.00mm.h5'), 'r')
ct_data = data['ct_data']
pet_data = data['pet_data']
y_data = data['label_data']
patient_ids = list(ct_data.keys())

def get_slices(data, patient_ids=patient_ids, use_pos_window=False):
    '''
    Arguments:
       shape: (voxel.shape[0], 100,100, 1)
       patient_ids: list of patient id numbers
       use_pos_window: boolean on whether or not you want positive only slices
    Returns:
       a tensor of the stacks of the slices with proper ordering 
    '''
    voxels = []
    for patient_id in patient_ids:
        voxel = data[patient_id].value
        
        if use_pos_window:
            window = POSITIVE_SLICES[patient_id]
            voxel = voxel[window[0]:window[1]]
        
        voxels += tf.split(tf.expand_dims(voxel, axis=3), voxel.shape[0])
    slices = tf.squeeze(tf.to_float(tf.stack(voxels)), [1])
    with tf.Session() as sess:
        return sess.run(slices)

#normalizes slices
def normalize(slices):
    '''
    Arguments:
        slices: numpy array or tensor of slices; tf ordering
    Returns:
        normalized numpy array/tensor
    '''
    with tf.Session() as sess:
        return sess.run(
            tf.map_fn(
                lambda img: tf.image.per_image_standardization(img), slices))

def get_patch_labels(y_slices, save = False):
    '''
    gets patch labels (centered pixels)
    y_slices: numpy array with (# of slices, l, w, channel) shape
    '''
    patches = tf.extract_image_patches(
        y_slices, PATCH_SIZE, PATCH_STRIDES, PATCH_RATES, PATCH_PADDING)
    square_patches = tf.reshape(patches, PATCH_SHAPE)
    center_pixels = square_patches[:, PATCH_HEIGHT // 2, PATCH_WIDTH // 2, :]
    center_patches = square_patches[:, PATCH_HEIGHT // 4 : -PATCH_HEIGHT // 4, PATCH_WIDTH // 4 : -PATCH_WIDTH // 4, :]
    y = tf.squeeze(tf.to_int32(tf.greater(center_pixels, 0)))
    y_patches = tf.to_int32(tf.greater(center_patches, 0))
    with tf.Session() as sess:
        y = sess.run(y)
    with tf.Session() as sess:
        y_patches = sess.run(y_patches)
    
    if save: 
        np.save('y_train.npy', y)
        np.save('y_patches_train.npy', y_patches)
    else:
        return y, y_patches

def get_patches(ct_slices, pet_slices, save = False, print_every=10): 
    '''
    Arguments:
        ct_slices: numpy array/tensor of CT slices; tf ordering
        pet_slices: ^^^ but for PET slices
        save: boolean on whether you want to save the outputted arrays
        print_every: integer when you want to print your progress when functon runs
    Returns:
        ct: numpy array of patches
        pet: numpy array of patches
    '''
    num_slices = ct_slices.shape[0]
        
    ct = []
    pet = []
        
    for i in range(num_slices):
            ct_slice = np.expand_dims(ct_slices[i], axis=0)
            pet_slice = np.expand_dims(pet_slices[i], axis=0)
            
            ct_patches = tf.extract_image_patches(
                ct_slice, PATCH_SIZE, PATCH_STRIDES, PATCH_RATES, PATCH_PADDING)
            pet_patches = tf.extract_image_patches(
                pet_slice, PATCH_SIZE, PATCH_STRIDES, PATCH_RATES, PATCH_PADDING)
            
            ct_square_patches = tf.reshape(ct_patches, PATCH_SHAPE)
            pet_square_patches = tf.reshape(pet_patches, PATCH_SHAPE)
            
            with tf.Session() as sess:
                ct_patches = sess.run(ct_square_patches)
            with tf.Session() as sess:
                pet_patches = sess.run(pet_square_patches)
            
            ct.append(ct_patches)
            pet.append(pet_patches)
            
            if (i + 1) % print_every == 0:
                print('{0} slices processed'.format((i+1)))
        
    ct = np.vstack(ct)
    pet = np.vstack(pet)
    
    if save: 
        np.save('ct_train.npy', ct)
        np.save('pet_train.npy', pet)
    else:
        return ct, pet

