# coding: utf-8
import numpy as np 
import os
import h5py
from sklearn.preprocessing import scale



def norm(a): 
    """normalizes input matrix between 0 and 1
    Args:
        a: numpy array
    Returns: 
        normalized numpy array
    """
    return (a - np.amin(a))/(np.amax(a)-np.amin(a))


def standardize_resize(a, reshape=[96,96]):
    """standardizes input and resizes
    Args: 
        a: input numpy array
        reshape: tuple of desired output shape
    Returns:
        Resized standardized numpy array
    """
    [r,c] = a.shape
    if reshape == [r,c]:
        z = np.expand_dims(scale(a),3)
        return z
    else:
        y = np.expand_dims(scale(np.resize(a, (96,96))),3)
        return y

#problem: if element, x, is 0<0.5<1, then rounding might fuck it up
def one_hot_seg(a):
    """one hot encodes a tensor for semantic segmentation
    Assumptions:
    --> np.amin(a) = 0
    Args:
        a is an input label 
    Returns:
        One hot encoded array with same dim as array, a.
    """
    #flat = np.sum(a,1).squeeze()
    onehot = np.clip(np.around((a)), a_min = 0, a_max = 1)
    return onehot

#testing this
def one_hot_seg_general(a):
    """one hot encodes a tensor for semantic segmentation
    Assumptions:
    --> np.amin(a) = 0
    Args:
        a is an input label 
    Returns:
        One hot encoded array with same dim as array, a.
    """
    #flat = np.sum(a,1).squeeze()
    onehot = np.clip(np.ceil((a)), a_min = 0, a_max = 1)
    return onehot


def remove_dead_slices(labels):
    """
    Args:
    labels: array of the labeled slices of shape(number_of_slices, rows, columns, number_of_channels)
    Returns:
    array of only labeled slices, array of indices
    """
    element_indices = list(np.nonzero(labels.squeeze()))
    labeled_indices = np.unique(element_indices[0])
    a = []
    for i in labeled_indices:
        a.append(labels[i])
    return np.asarray(a), labeled_indices

def open_data(p_list, file_path, depth = 175):
    """Opens the CT Scan data for preprocessing from a h5py file.
    
    Args:
        p_list: list of patient names
        file_path: file path to the h5py file: lab_petct_vox_5.00mm.h5
        depth: number of slices per patient (assumes uniformity)
    Returns:
        A list with all of the np slices
    """
    data = h5py.File(file_path, 'r')
    ct_data = data['ct_data']
    return [np.expand_dims(np.resize(np.array(ct_data[p_id][i]),(96,96)),3) for p_id in p_list for i in range(depth)]

def open_l_data(p_list, file_path, depth = 175):
    """Opens the CT Scan data for preprocessing from a h5py file.
    
    Args:
        p_list: list of patient names
        file_path: file path to the h5py file: lab_petct_vox_5.00mm.h5
        depth: number of slices per patient (assumes uniformity)
    Returns:
        A list with all of the np slices
    """
    data = h5py.File(file_path, 'r')
    label_data = data['label_data']
    return [np.expand_dims(np.resize(np.array(label_data[p_id][i]),(96,96)),3) for p_id in p_list for i in range(depth)]






