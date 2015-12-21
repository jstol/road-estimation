import numpy as np

superpixels = [100, 1000, 5000, 10000, 15000, 20000]
data_sets = ["train", "valid", "test"]

for sp in superpixels:
	for data_set in data_sets:
		x = np.load("superpixel_data/{0}_examples_{1}sp.npz".format(data_set, sp))
		np.savez("oracle_predictions/{0}sp_{1}.npz".format(sp, data_set), predictions=x['targets'])
