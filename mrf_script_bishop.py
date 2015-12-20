#!/usr/bin/env python2.7
# -*- coding: utf8 -*-
from __future__ import absolute_import, print_function, unicode_literals
"""defines classes that encapsulates the sci-kit learn ML algorithms"""

#general
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
import math
import random

#image processing
from skimage.io import imread_collection, imread, imshow, imsave
from skimage import img_as_float, img_as_uint
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2

#markov random field
from markovrandomfield_bishop import pixelmap




print('==============================')
print('Testing')
print('==============================')

print('==============================')
print('Load prediction image')
print('==============================')

image_type = np.uint16
max_val = np.iinfo(image_type).max
#orig_photo = imread('example-predictions/original-photo/umm_000080.png')
prediction_image = imread('example-predictions/encoded/umm_road_000080.png', grey=True).astype(image_type)
image_height, image_width = prediction_image.shape[0], prediction_image.shape[1]
image_pixel_priors = prediction_image/(float(max_val))

print('========================================')
print('Apply pre-processing to prediction image')
print('=========================================')

print('Gaussian Blur')
image_pixel_priors = gaussian_filter(image_pixel_priors, 5)

# print('Bilateral Filter')
# image_pixel_priors = denoise_bilateral(image_pixel_priors, sigma_range=0.05, sigma_spatial=15)

image_pixel_priors_flat = image_pixel_priors.ravel()

#
print('Testing - Max and Min Value')
print(np.max(image_pixel_priors_flat))
print(np.min(image_pixel_priors_flat))


#===================================
#script for initiating the MRF class
#===================================

print('============================')
print('Initializing MRF:')
print('============================')

predicted_labels = pixelmap()

predicted_labels.load_superpixel_classifier_predictions(image_pixel_priors_flat, prediction_image.shape[0], prediction_image.shape[1])
predicted_labels.set_conn_energy(0.5) #this is required to set the strength of connections ()
predicted_labels.init_energy()

print('============================')
print('Displaying initial state info:')
print('============================')


updated_predictions = np.reshape(predicted_labels.pixel_labels, [prediction_image.shape[0], prediction_image.shape[1]])

imsave('testing-orig-predictions.png', prediction_image)

imsave('testing-gaussian-filtered-predictions.png', image_pixel_priors)

imsave('testing-updated-predictions-start.png', updated_predictions)

print('Size of uncertain region: %f' %(len(predicted_labels.uncertain_pixel_list)))

print('Initial energy:')
print(predicted_labels.total_energy)



print('============================')
print('MCMC Iterations ... ')
print('============================')

for t in [2.0, 1.5, 1.0, 0.5, 0.2]:

	print('Temperature: %f' %(t))

	predicted_labels.mcmc_rand_update(1/t)
	print(predicted_labels.total_energy)
	predicted_labels.mcmc_block_flip_update(1/t)
	print(predicted_labels.total_energy)
	predicted_labels.mcmc_rand_update(1/t)
	print(predicted_labels.total_energy)
	predicted_labels.mcmc_update(1/t)
	print(predicted_labels.total_energy)

print('============================')
print('Saving result to file:')
print('============================')

updated_predictions = np.reshape(predicted_labels.pixel_labels, [prediction_image.shape[0], prediction_image.shape[1]])

imsave('testing-updated-predictions-end.png', updated_predictions)

