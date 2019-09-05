# %%
# import
import os
import sys
import numpy as np
from skimage.io import imread, imsave

# %%
# setting
IMG_PATH = 'Aberdeen'

test_img = ['1.jpg', '45.jpg', '108.jpg', '256.jpg', '375.jpe']

# number of principal component used
k = 5

# %%
# Image processing
def process(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)


# %%
# Load Data
filelist = os.listdir(IMG_PATH)
filelist

img_shape = imread(os.path.join(IMG_PATH, filelist[0])).shape

img_data = []

for file in filelist:
    if file != '.DS_Store':
        tmp = imread(os.path.join(IMG_PATH, file))
        img_data.append(tmp.flatten())

# %%
# Normalize & SVD
training_data = np.array(img_data).astype(np.float32)
mean = np.mean(training_data, 0)
training_data -= mean
u, s, v = np.linalg.svd(training_data, full_matrices=False)
u.shape
s.shape
v.shape
# %%
# compression and reconstruction
x = test_img[0]

picked_img = imread(os.path.join(IMG_PATH, x))
X = picked_img.flatten().astype('float32')
X -= mean
X.shape
# Compression
weight = np.array([u[i].dot(s) for i in range(k)])
weight.shape
# Reconstruction
mean.shape
X.dot(weight).shape
reconstruct = process(weight.dot(v) + mean)
reconstruct.shape
imsave(x[:-4] + '_reconstruction.jpg', reconstruct.reshape(img_shape))
