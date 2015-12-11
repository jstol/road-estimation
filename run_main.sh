#!/usr/bin/env bash
alg=$1
params=$2
param_configuration_subdir=$3
superpixels_set=(100 1000 5000 10000 15000 20000)
=======
start_date=$4
superpixels_set=(100 1000 5000 10000 15000 20000)
>>>>>>> 2d21afad91618bf7dd4740bc10e06a34926f8b59

run_dir="results/${alg}"
results_dir="${run_dir}/${param_configuration_subdir}"

echo "====================================="
echo "Running for ${alg} - ${params}"
echo "====================================="
for i in "${superpixels_set[@]}"
do
	# Train, make predictions
	echo "----------------"
	echo "${i} superpixels"
	echo "----------------"
	python main.py -m "${alg}" -o "${results_dir}/test_results/${alg}_${i}sp_superpixel_level.txt" -p "${params}"\
		--train-data "superpixel_data/train_examples_${i}sp.npz" --train-predictions-output "${results_dir}/predictions/${alg}_${i}sp_train.npz" \
		--valid-data "superpixel_data/valid_examples_${i}sp.npz" --valid-predictions-output "${results_dir}/predictions/${alg}_${i}sp_valid.npz" \
		--model-file "${results_dir}/models/${alg}_${i}sp.npz" \
		--summary-file "${run_dir}/superpixel_report_${start_date}.csv"

	# Encode predictions and get pixel-level results
	pixel_test_results_file="${results_dir}/test_results/${alg}_${i}sp_pixel_level.txt"
	printf "PIXEL LEVEL RESULTS\n\n" > "${pixel_test_results_file}"
	printf "===================\n" >> "${pixel_test_results_file}"
	printf "TRAINING\n" >> "${pixel_test_results_file}"
	printf "===================\n" >> "${pixel_test_results_file}"

	echo "Encoding training predictions"
	python encode_predictions.py \
		-m "superpixel_data/train_superpixels_map_${i}sp.npz" \
		-i "${results_dir}/predictions/${alg}_${i}sp_train.npz" \
		-e "kit/data_road/training/divided_data/train/image_2" \
		-o "${results_dir}/prediction_images/${i}sp/train/encoded" \
		-ov "${results_dir}/prediction_images/${i}sp/train/encoded_overlay" >/dev/null 2>&1
		# --generate-overlay \

	python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/train/encoded/" "kit/data_road/training/divided_data/train" "${run_dir}/pixel_report_${start_date}.csv" "knn" "${params}" "train" >> "${pixel_test_results_file}"

	printf "\n===================\n" >> "${pixel_test_results_file}"
	printf "VALID\n" >> "${pixel_test_results_file}"
	printf "===================\n" >> "${pixel_test_results_file}"

	echo "Encoding valid predictions"
	python encode_predictions.py \
		-m "superpixel_data/valid_superpixels_map_${i}sp.npz" \
		-i "${results_dir}/predictions/${alg}_${i}sp_valid.npz" \
		-e "kit/data_road/training/divided_data/valid/image_2" \
		-o "${results_dir}/prediction_images/${i}sp/valid/encoded" \
		-ov "${results_dir}/prediction_images/${i}sp/valid/encoded_overlay" >/dev/null 2>&1
		# --generate-overlay \

	python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/valid/encoded/" "kit/data_road/training/divided_data/valid" "${run_dir}/pixel_report_${start_date}.csv" "knn" "${params}" "valid" >> "${pixel_test_results_file}"

	# Write param config to a file
	echo ${params} > "${results_dir}/params_configuration.txt"
done
