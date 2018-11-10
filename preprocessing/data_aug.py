import tensorflow as tf
from skimage import transform

def random_crop_and_pad_image_and_labels(image, labels, size):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, labels], axis=2)
  image_shape = tf.shape(image)
  combined_pad = tf.image.pad_to_bounding_box(
      combined, 0, 0,
      tf.maximum(size[0], image_shape[0]),
      tf.maximum(size[1], image_shape[1]))
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  combined_crop = tf.random_crop(
      combined_pad,
      size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
  return (combined_crop[:, :, :last_image_dim],
          combined_crop[:, :, last_image_dim:])
# 2D
def data_aug(image,label,angle=30,resize_rate=0.9):
    '''
    https://www.kaggle.com/shenmbsw/data-augmentation-and-tensorflow-u-net
    '''
    flip = random.randint(0, 1)
    size = image.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angle = random.random()/180*np.pi*angle
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angle)
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    # Randomly corpping image frame
    image = image[w_s:w_s+size,h_s:h_s+size,:]
    label = label[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
    return image, label

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates

# Function to distort image
def elastic_transform(image, alpha=2000, sigma=40, alpha_affine=40, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    for i in range(shape[2]):
        image[:,:,i] = cv2.warpAffine(image[:,:,i], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    image = image.reshape(shape)

    blur_size = int(4*sigma) | 1

    dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    def_img = np.zeros_like(image)
    for i in range(shape[2]):
        def_img[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape_size)

    return def_img


def salt_pepper_noise(image, salt=0.2, amount=0.004):
    row, col, chan = image.shape
    num_salt = np.ceil(amount * row * salt)
    num_pepper = np.ceil(amount * row * (1.0 - salt))

    for n in range(chan//2): # //2 so we don't augment the mask
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[0:2]]
        image[coords[0], coords[1], n] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[0:2]]
        image[coords[0], coords[1], n] = 0

    return image
