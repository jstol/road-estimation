import matplotlib.pyplot as plot
from skimage.io import imread_collection, imread
import numpy as np

maps = np.load('pixelmaps.npz')
mask = maps['um_000000']
# examples = np.load('examples.npz')
# targets = examples['targets']

predictions = np.load('example-predictions.npz')

i = imread('kit/data_road/training/image_2/um_000000.png')

#for j in predictions.nonzero()[0]:
for j in xrange(np.max(mask)):
	i[mask == j] = 255

plot.imshow(i)
plot.show()
