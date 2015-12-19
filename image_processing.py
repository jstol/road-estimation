

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
import math

from skimage.io import imread_collection, imread, imshow, imsave
from skimage import img_as_float, img_as_uint
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

from skimage import measure
from scipy import ndimage as ndi

from skimage import feature

from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte, img_as_uint

from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage.util.shape import view_as_windows
from skimage.util.montage import montage2d


import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure

import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data

from skimage import data, img_as_float
from skimage import exposure

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float



image_type = np.uint16
max_val = np.iinfo(image_type).max
road_example = imread('example-predictions/original-photo/um_000042.png')
road_example_hsv = color.rgb2hsv(road_example)
imshow(road_example_hsv)
road_example_grey = color.rgb2grey(road_example)
plt.show()
imshow(road_example)
image = img_as_ubyte(road_example_grey)
plt.show()


# Load an example image
img = img_as_ubyte(road_example_grey)

# Global equalize
img_rescale = exposure.equalize_hist(img)

# Equalization
selem = disk(30)
img_eq = rank.equalize(img, selem=selem)


# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 3), dtype=np.object)
axes[0,0] = plt.subplot(2, 3, 1, adjustable='box-forced')
axes[0,1] = plt.subplot(2, 3, 2, sharex=axes[0,0], sharey=axes[0,0], adjustable='box-forced')
axes[0,2] = plt.subplot(2, 3, 3, sharex=axes[0,0], sharey=axes[0,0], adjustable='box-forced')
axes[1,0] = plt.subplot(2, 3, 4)
axes[1,1] = plt.subplot(2, 3, 5)
axes[1,2] = plt.subplot(2, 3, 6)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')
ax_hist.set_ylabel('Number of pixels')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
ax_img.set_title('Global equalise')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
ax_img.set_title('Local equalize')
ax_cdf.set_ylabel('Fraction of total intensity')


# prevent overlap of y-axis labels
fig.subplots_adjust(wspace=0.4)
plt.show()


#def run_HOG():
image = img_rescale

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()


#entropy
image = img_rescale
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Image')
ax0.axis('off')
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(entropy(image, disk(5)), cmap=plt.cm.jet)
ax1.set_title('Entropy')
ax1.axis('off')
fig.colorbar(img1, ax=ax1)

plt.show()



#edge detector
# Compute the Canny filter for two values of sigma
im = img_rescale
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.jet)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()

#segementation
fill_road = ndi.binary_fill_holes(edges1)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_road, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Filling the holes')

plt.show()

markers = np.zeros_like(fill_road)
markers[50, 500] = 1
markers[300, 10] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
ax.axis('off')
ax.set_title('markers')

plt.show()

#find local max
im = img_as_float(road_example_grey)

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(im, size=20, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(im, min_distance=20)

# display results
fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax1, ax2, ax3 = ax.ravel()
ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Original')

ax2.imshow(image_max, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Maximum filter')

ax3.imshow(im, cmap=plt.cm.gray)
ax3.autoscale(False)
ax3.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax3.axis('off')
ax3.set_title('Peak local max')

fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)

plt.show()






def run_gabor():

	def compute_feats(image, kernels):
	    feats = np.zeros((len(kernels), 2), dtype=np.double)
	    for k, kernel in enumerate(kernels):
	        filtered = ndi.convolve(image, kernel, mode='wrap')
	        feats[k, 0] = filtered.mean()
	        feats[k, 1] = filtered.var()
	    return feats


	def match(feats, ref_feats):
	    min_error = np.inf
	    min_i = None
	    for i in range(ref_feats.shape[0]):
	        error = np.sum((feats - ref_feats[i, :])**2)
	        if error < min_error:
	            min_error = error
	            min_i = i
	    return min_i


	# prepare filter bank kernels
	kernels = []
	for theta in range(4):
	    theta = theta / 4. * np.pi
	    for sigma in (1, 3):
	        for frequency in (0.05, 0.25):
	            kernel = np.real(gabor_kernel(frequency, theta=theta,
	                                          sigma_x=sigma, sigma_y=sigma))
	            kernels.append(kernel)


	shrink = (slice(0, None, 3), slice(0, None, 3))
	brick = road_example_grey
	grass = road_example_grey
	wall = road_example_grey
	image_names = ('brick', 'grass', 'wall')
	images = (brick, grass, wall)

	# prepare reference features
	ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
	ref_feats[0, :, :] = compute_feats(brick, kernels)
	ref_feats[1, :, :] = compute_feats(grass, kernels)
	ref_feats[2, :, :] = compute_feats(wall, kernels)

	print('Rotated images matched against references using Gabor filter banks:')

	print('original: brick, rotated: 30deg, match result: ', end='')
	feats = compute_feats(ndi.rotate(brick, angle=190, reshape=False), kernels)
	print(image_names[match(feats, ref_feats)])

	print('original: brick, rotated: 70deg, match result: ', end='')
	feats = compute_feats(ndi.rotate(brick, angle=70, reshape=False), kernels)
	print(image_names[match(feats, ref_feats)])

	print('original: grass, rotated: 145deg, match result: ', end='')
	feats = compute_feats(ndi.rotate(grass, angle=145, reshape=False), kernels)
	print(image_names[match(feats, ref_feats)])


	def power(image, kernel):
	    # Normalize images for better comparison.
	    image = (image - image.mean()) / image.std()
	    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
	                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

	# Plot a selection of the filter bank kernels and their responses.
	results = []
	kernel_params = []
	for theta in (0, 1):
	    theta = theta / 4. * np.pi
	    for frequency in (0.1, 0.4):
	        kernel = gabor_kernel(frequency, theta=theta)
	        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
	        kernel_params.append(params)
	        # Save kernel and the power image for each image
	        results.append((kernel, [power(img, kernel) for img in images]))

	fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
	plt.gray()

	fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

	axes[0][0].axis('off')

	# Plot original images
	for label, img, ax in zip(image_names, images, axes[0][1:]):
	    ax.imshow(img)
	    ax.set_title(label, fontsize=9)
	    ax.axis('off')

	for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
	    # Plot Gabor kernel
	    ax = ax_row[0]
	    ax.imshow(np.real(kernel), interpolation='nearest')
	    ax.set_ylabel(label, fontsize=7)
	    ax.set_xticks([])
	    ax.set_yticks([])

	    # Plot Gabor responses with the contrast normalized for each filter
	    vmin = np.min(powers)
	    vmax = np.max(powers)
	    for patch, ax in zip(powers, ax_row[1:]):
	        ax.imshow(patch, vmin=vmin, vmax=vmax)
	        ax.axis('off')

	plt.show()


def run_hist_renormalization():
	matplotlib.rcParams['font.size'] = 8


	def plot_img_and_hist(img, axes, bins=256):
	    """Plot an image along with its histogram and cumulative histogram.

	    """
	    img = img_as_float(img)
	    ax_img, ax_hist = axes
	    ax_cdf = ax_hist.twinx()

	    # Display image
	    ax_img.imshow(img, cmap=plt.cm.gray)
	    ax_img.set_axis_off()
	    ax_img.set_adjustable('box-forced')

	    # Display histogram
	    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
	    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
	    ax_hist.set_xlabel('Pixel intensity')
	    ax_hist.set_xlim(0, 1)
	    ax_hist.set_yticks([])

	    # Display cumulative distribution
	    img_cdf, bins = exposure.cumulative_distribution(img, bins)
	    ax_cdf.plot(bins, img_cdf, 'r')
	    ax_cdf.set_yticks([])

	    return ax_img, ax_hist, ax_cdf


	# Load an example image
	img = road_example_grey

	# Contrast stretching
	p2, p98 = np.percentile(img, (2, 98))
	img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

	# Equalization
	img_eq = exposure.equalize_hist(img)

	# Adaptive Equalization
	img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

	# Display results
	fig = plt.figure(figsize=(8, 5))
	axes = np.zeros((2,4), dtype=np.object)
	axes[0,0] = fig.add_subplot(2, 4, 1)
	for i in range(1,4):
	    axes[0,i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
	for i in range(0,4):
	    axes[1,i] = fig.add_subplot(2, 4, 5+i)

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
	ax_img.set_title('Low contrast image')

	y_min, y_max = ax_hist.get_ylim()
	ax_hist.set_ylabel('Number of pixels')
	ax_hist.set_yticks(np.linspace(0, y_max, 5))

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
	ax_img.set_title('Contrast stretching')

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
	ax_img.set_title('Histogram equalization')

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
	ax_img.set_title('Adaptive equalization')

	ax_cdf.set_ylabel('Fraction of total intensity')
	ax_cdf.set_yticks(np.linspace(0, 1, 5))

	# prevent overlap of y-axis labels
	fig.subplots_adjust(wspace=0.4)
	plt.show()

def run_gabor_filters2():
	np.random.seed(42)

	patch_shape = 8, 8
	n_filters = 49

	astro = road_example_grey

	# -- filterbank1 on original image
	patches1 = view_as_windows(astro, patch_shape)
	patches1 = patches1.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
	fb1, _ = kmeans2(patches1, n_filters, minit='points')
	fb1 = fb1.reshape((-1,) + patch_shape)
	fb1_montage = montage2d(fb1, rescale_intensity=True)

	# -- filterbank2 LGN-like image
	astro_dog = ndi.gaussian_filter(astro, .5) - ndi.gaussian_filter(astro, 1)
	patches2 = view_as_windows(astro_dog, patch_shape)
	patches2 = patches2.reshape(-1, patch_shape[0] * patch_shape[1])[::8]
	fb2, _ = kmeans2(patches2, n_filters, minit='points')
	fb2 = fb2.reshape((-1,) + patch_shape)
	fb2_montage = montage2d(fb2, rescale_intensity=True)

	# --
	fig, axes = plt.subplots(2, 2, figsize=(7, 6))
	ax0, ax1, ax2, ax3 = axes.ravel()

	ax0.imshow(astro, cmap=plt.cm.gray)
	ax0.set_title("Image (original)")

	ax1.imshow(fb1_montage, cmap=plt.cm.gray, interpolation='nearest')
	ax1.set_title("K-means filterbank (codebook)\non original image")

	ax2.imshow(astro_dog, cmap=plt.cm.gray)
	ax2.set_title("Image (LGN-like DoG)")

	ax3.imshow(fb2_montage, cmap=plt.cm.gray, interpolation='nearest')
	ax3.set_title("K-means filterbank (codebook)\non LGN-like DoG image")

	for ax in axes.ravel():
	    ax.axis('off')

	fig.subplots_adjust(hspace=0.3)
	plt.show()


def run_glcm(): #does not work
	PATCH_SIZE = 21

	# open the camera image
	image = road_example_grey

	# select some patches from grassy areas of the image
	grass_locations = [(474, 291), (440, 433), (466, 18), (462, 236)]
	grass_patches = []
	for loc in grass_locations:
	    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
	                               loc[1]:loc[1] + PATCH_SIZE])

	# select some patches from sky areas of the image
	sky_locations = [(54, 48), (21, 233), (90, 380), (195, 330)]
	sky_patches = []
	for loc in sky_locations:
	    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
	                             loc[1]:loc[1] + PATCH_SIZE])

	# compute some GLCM properties each patch
	xs = []
	ys = []
	for patch in (grass_patches + sky_patches):
	    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
	    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
	    ys.append(greycoprops(glcm, 'correlation')[0, 0])

	# create the figure
	fig = plt.figure(figsize=(8, 8))

	# display original image with locations of patches
	ax = fig.add_subplot(3, 2, 1)
	ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
	          vmin=0, vmax=255)
	for (y, x) in grass_locations:
	    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
	for (y, x) in sky_locations:
	    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
	ax.set_xlabel('Original Image')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.axis('image')

	# for each patch, plot (dissimilarity, correlation)
	ax = fig.add_subplot(3, 2, 2)
	ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
	        label='Grass')
	ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
	        label='Sky')
	ax.set_xlabel('GLCM Dissimilarity')
	ax.set_ylabel('GLVM Correlation')
	ax.legend()

	# display the image patches
	for i, patch in enumerate(grass_patches):
	    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
	    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
	              vmin=0, vmax=255)
	    ax.set_xlabel('Grass %d' % (i + 1))

	for i, patch in enumerate(sky_patches):
	    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
	    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
	              vmin=0, vmax=255)
	    ax.set_xlabel('Sky %d' % (i + 1))


	# display the patches and plot
	fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
	plt.show()

matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def run_correction():
	# Load an example image
	img = road_example_grey

	# Gamma
	gamma_corrected = exposure.adjust_gamma(img, 2)

	# Logarithmic
	logarithmic_corrected = exposure.adjust_log(img, 1)

	# Display results
	fig = plt.figure(figsize=(8, 5))
	axes = np.zeros((2,3), dtype=np.object)
	axes[0, 0] = plt.subplot(2, 3, 1, adjustable='box-forced')
	axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0], adjustable='box-forced')
	axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0], adjustable='box-forced')
	axes[1, 0] = plt.subplot(2, 3, 4)
	axes[1, 1] = plt.subplot(2, 3, 5)
	axes[1, 2] = plt.subplot(2, 3, 6)

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
	ax_img.set_title('Low contrast image')

	y_min, y_max = ax_hist.get_ylim()
	ax_hist.set_ylabel('Number of pixels')
	ax_hist.set_yticks(np.linspace(0, y_max, 5))

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(gamma_corrected, axes[:, 1])
	ax_img.set_title('Gamma correction')

	ax_img, ax_hist, ax_cdf = plot_img_and_hist(logarithmic_corrected, axes[:, 2])
	ax_img.set_title('Logarithmic correction')

	ax_cdf.set_ylabel('Fraction of total intensity')
	ax_cdf.set_yticks(np.linspace(0, 1, 5))

	# prevent overlap of y-axis labels
	fig.subplots_adjust(wspace=0.4)
	plt.show()


matplotlib.rcParams['font.size'] = 9


def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[img.dtype.type]
    ax_hist.set_xlim(xmin, xmax)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf