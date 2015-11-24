#!/usr/bin/env bash
echo "----------------------------"
echo "Running for 1000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/image_2/*.png -t kit/data_road/training/gt_image_2 -m 'superpixels_map_1000sp.npz' -e 'examples_1000sp.npz' -n 1000
echo "----------------------------"
echo "Running for 5000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/image_2/*.png -t kit/data_road/training/gt_image_2 -m 'superpixels_map_5000sp.npz' -e 'examples_5000sp.npz' -n 5000
echo "----------------------------"
echo "Running for 10000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/image_2/*.png -t kit/data_road/training/gt_image_2 -m 'superpixels_map_10000sp.npz' -e 'examples_10000sp.npz' -n 10000
echo "----------------------------"
echo "Running for 15000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/image_2/*.png -t kit/data_road/training/gt_image_2 -m 'superpixels_map_15000sp.npz' -e 'examples_15000sp.npz' -n 15000
echo "----------------------------"
echo "Running for 20000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/image_2/*.png -t kit/data_road/training/gt_image_2 -m 'superpixels_map_20000sp.npz' -e 'examples_20000sp.npz' -n 20000
