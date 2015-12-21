#!/usr/bin/env bash
alg="oracle"
params="ORACLE"
start_date=$4
superpixels_set=(100 1000 5000 10000 15000 20000)

run_dir="oracle_results"
results_dir="${run_dir}/data"

echo "====================================="
echo "Running for ${ensemble_method} ${alg} - ${params}"
echo "====================================="
for i in "${superpixels_set[@]}"
do
	# Train, make predictions
	echo "----------------"
	echo "${i} superpixels"
	echo "----------------"

	# Encode predictions and get pixel-level results
	pixel_test_results_file="${results_dir}/test_results/${ensemble_method}${alg}_${i}sp_pixel_level.txt"
	printf "PIXEL LEVEL RESULTS\n\n" > "${pixel_test_results_file}"
	printf "===================\n" >> "${pixel_test_results_file}"
	printf "TRAINING\n" >> "${pixel_test_results_file}"
	printf "===================\n" >> "${pixel_test_results_file}"

	echo "Encoding training predictions"
	python encode_predictions.py \
		-m "superpixel_data/train_superpixels_map_${i}sp.npz" \
		-i "oracle_predictions/${i}sp_train.npz" \
		-e "kit/data_road/training/divided_data/train/image_2" \
		-o "${results_dir}/prediction_images/${i}sp/train/encoded" \
		-ov "${results_dir}/prediction_images/${i}sp/train/encoded_overlay" >/dev/null 2>&1
		# --generate-overlay \

	python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/train/encoded/" "kit/data_road/training/divided_data/train" "${run_dir}/pixel_report_${start_date}.csv" "${ensemble_method}${alg}" "${params}" "train" "${i}" >> "${pixel_test_results_file}"

	printf "\n===================\n" >> "${pixel_test_results_file}"
	printf "VALID\n" >> "${pixel_test_results_file}"
	printf "===================\n" >> "${pixel_test_results_file}"

	echo "Encoding valid predictions"
	python encode_predictions.py \
		-m "superpixel_data/valid_superpixels_map_${i}sp.npz" \
		-i "oracle_predictions/${i}sp_valid.npz" \
		-e "kit/data_road/training/divided_data/valid/image_2" \
		-o "${results_dir}/prediction_images/${i}sp/valid/encoded" \
		-ov "${results_dir}/prediction_images/${i}sp/valid/encoded_overlay" >/dev/null 2>&1
		# --generate-overlay \

	python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/valid/encoded/" "kit/data_road/training/divided_data/valid" "${run_dir}/pixel_report_${start_date}.csv" "${ensemble_method}${alg}" "${params}" "valid" "${i}" >> "${pixel_test_results_file}"

	# Write param config to a file
	echo ${params} > "${results_dir}/params_configuration.txt"
done
