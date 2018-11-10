#channels_last gnerator
import keras
import SimpleITK as sitk
from SimpleITK import GetArrayFromImage
from random import randint
import numpy as np
from glob import glob
import os

from data_aug import random_crop_and_pad_image_and_labels

class MedicalPatchwiseSequence2D(keras.utils.Sequence):
    '''
    For generating 2D thread-safe data in keras. (no preprocessing and channels_last)

    Attributes:
      x_set, y_set: list of paths to the training and label data (.nii) respectively
      batch_size: int of desired number images per epoch
      patch_shape: shape of the desired window size without the channel
      patch_overlap: desired overlap
      n_channels: <-
    '''
    def __init__(self, x_set, y_set, batch_size, image_shape = (155,240,240), patch_shape = (128,128,128), patch_overlap = 16, n_channels = 4, shuffle = True):
        # lists of paths to images
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        self.n_channels = n_channels
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        Defines the fetching and on-the-fly preprocessing of data.
        '''
        # file names
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = self.__data_gen(batch_x, batch_y)
        return (X, y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # why does this work; self.filenames
        self.idx = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.idx)

    def __data_gen(self, batch_x, batch_y):
        '''generates the data'''
        patches_x = []
        patches_y = []
        for file_x, file_y in zip(batch_x, batch_y):
          ### CHECK IF IT LOADS 1 OR MULTIPLE FILES (MAKE IT COMPATIBLE)
          sitk_image, sitk_label = sitk.ReadImage(file_x), sitk.ReadImage(file_y)
          # # for binary only
          # image_shape = read_x.GetSize()[::-1]

            ##### NEEDS TO HAVE THE SLICE RANDOMIZED
          x_train = np.expand_dims(GetArrayFromImage(sitk_image), -1)
          y_train = np.expand_dims(GetArrayFromImage(sitk_label), -1)

          # randomizing which slice to pick
          slice_idx = randint(0, x_train.shape[0])
          x_train = x_train[slice_idx-1:slice_idx]
          y_train = y_train[slice_idx-1:slice_idx]
          patches_x.append(x_train), patches_y.append(y_train)

        return [np.vstack(patches_x), np.vstack(patches_y)], [np.vstack(patches_y), np.vstack(patches_y) * np.vstack(patches_x)]

    def test_batch_order(self, idx):
        '''for testing purposes'''
        # file names
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

### For quick testing
# if __name__ == "__main__":
#     # getting paths for data
#     os.chdir('C:\\Users\\Joseph')
#     local_train_path = "MSD\\MSD_raw\\Task02_Heart\\imagesTr\\"
#     local_label_path = 'MSD\\MSD_raw\\Task02_Heart\\labelsTr\\'
#     train_paths = glob(local_train_path + '**.nii', recursive = True)
#     mask_paths = glob(local_label_path + '**.nii', recursive = True)
#
#     # data_dict = {'data': train_paths, 'seg': mask_paths}
#
#     seq = MedicalPatchwiseSequence2D(train_paths, mask_paths, batch_size = 2,
#                                     image_shape = (320, 320, 1), patch_shape = None,
#                                     n_channels = 1
#                                     )
#     x, y = seq.__getitem__(5)
#     print(x[0].shape)

        # train_model.fit_generator(generator = seq)
