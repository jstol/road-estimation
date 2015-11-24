#!/usr/bin/env bash
echo "----------------------------"
echo "Running for 1000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/divided_data/train/image_2/*.png -t kit/data_road/training/divided_data/train/gt_image_2 -m 'train_superpixels_map_1000sp.npz' -e 'train_examples_1000sp.npz' -n 1000
python precompute.py -i kit/data_road/training/divided_data/valid/image_2/*.png -t kit/data_road/training/divided_data/valid/gt_image_2 -m 'valid_superpixels_map_1000sp.npz' -e 'valid_examples_1000sp.npz' -n 1000
python precompute.py -i kit/data_road/training/divided_data/test/image_2/*.png -t kit/data_road/training/divided_data/test/gt_image_2 -m 'test_superpixels_map_1000sp.npz' -e 'test_examples_1000sp.npz' -n 1000
echo "----------------------------"
echo "Running for 5000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/divided_data/train/image_2/*.png -t kit/data_road/training/divided_data/train/gt_image_2 -m 'train_superpixels_map_5000sp.npz' -e 'train_examples_5000sp.npz' -n 5000
python precompute.py -i kit/data_road/training/divided_data/valid/image_2/*.png -t kit/data_road/training/divided_data/valid/gt_image_2 -m 'valid_superpixels_map_5000sp.npz' -e 'valid_examples_5000sp.npz' -n 5000
python precompute.py -i kit/data_road/training/divided_data/test/image_2/*.png -t kit/data_road/training/divided_data/test/gt_image_2 -m 'test_superpixels_map_5000sp.npz' -e 'test_examples_5000sp.npz' -n 5000
echo "----------------------------"
echo "Running for 10000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/divided_data/train/image_2/*.png -t kit/data_road/training/divided_data/train/gt_image_2 -m 'train_superpixels_map_10000sp.npz' -e 'train_examples_10000sp.npz' -n 10000
python precompute.py -i kit/data_road/training/divided_data/valid/image_2/*.png -t kit/data_road/training/divided_data/valid/gt_image_2 -m 'valid_superpixels_map_10000sp.npz' -e 'valid_examples_10000sp.npz' -n 10000
python precompute.py -i kit/data_road/training/divided_data/test/image_2/*.png -t kit/data_road/training/divided_data/test/gt_image_2 -m 'test_superpixels_map_10000sp.npz' -e 'test_examples_10000sp.npz' -n 10000
echo "----------------------------"
echo "Running for 15000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/divided_data/train/image_2/*.png -t kit/data_road/training/divided_data/train/gt_image_2 -m 'train_superpixels_map_15000sp.npz' -e 'train_examples_15000sp.npz' -n 15000
python precompute.py -i kit/data_road/training/divided_data/valid/image_2/*.png -t kit/data_road/training/divided_data/valid/gt_image_2 -m 'valid_superpixels_map_15000sp.npz' -e 'valid_examples_15000sp.npz' -n 15000
python precompute.py -i kit/data_road/training/divided_data/test/image_2/*.png -t kit/data_road/training/divided_data/test/gt_image_2 -m 'test_superpixels_map_15000sp.npz' -e 'test_examples_15000sp.npz' -n 15000
echo "----------------------------"
echo "Running for 20000 superpixels"
echo "----------------------------"
python precompute.py -i kit/data_road/training/divided_data/train/image_2/*.png -t kit/data_road/training/divided_data/train/gt_image_2 -m 'train_superpixels_map_20000sp.npz' -e 'train_examples_20000sp.npz' -n 20000
python precompute.py -i kit/data_road/training/divided_data/valid/image_2/*.png -t kit/data_road/training/divided_data/valid/gt_image_2 -m 'valid_superpixels_map_20000sp.npz' -e 'valid_examples_20000sp.npz' -n 20000
python precompute.py -i kit/data_road/training/divided_data/test/image_2/*.png -t kit/data_road/training/divided_data/test/gt_image_2 -m 'test_superpixels_map_20000sp.npz' -e 'test_examples_20000sp.npz' -n 20000
