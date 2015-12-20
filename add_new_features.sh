#!/usr/bin/env bash
superpixels_set=(100 1000 5000 10000 15000 20000)
data_sets=('train' 'valid' 'test')

for i in "${superpixels_set[@]}"
do
	echo "====================================="
	echo "SUPERPIXEL SET ${i}"
	echo "====================================="

	for data_set in "${data_sets[@]}"
	do
		echo "-------------------------------------"
		echo "DATA SET ${data_set}"
		echo "-------------------------------------"

		python extra_features.py \
			-m "superpixel_data/${data_set}_superpixels_map_${i}sp.npz" \
			-i "superpixel_data/${data_set}_examples_${i}sp.npz" \
			-e "kit/data_road/training/divided_data/${data_set}/image_2/" \
			-o "extra_feature_matrices"
	done
done
