#!/usr/bin/env bash
alg=$1
params=$2
param_configuration_subdir=$3
start_date=$4
ensemble_method=$5
# superpixels_set=(100 1000 5000 10000)
superpixels_set=(5000)

run_dir="FINAL_results/${ensemble_method}${alg}"
results_dir="${run_dir}/${param_configuration_subdir}"

echo "====================================="
echo "Running for ${ensemble_method} ${alg} - ${params}"
echo "====================================="
for i in "${superpixels_set[@]}"
do
	# Train, make predictions
	echo "----------------"
	echo "${i} superpixels"
	echo "----------------"
	python final_main.py -m "${alg}" -e "${ensemble_method}" -o "${results_dir}/test_results/${ensemble_method}${alg}_${i}sp_superpixel_level.txt" -p "${params}"\
		--train-data "superpixel_data/train_examples_${i}sp.npz" --train-predictions-output "${results_dir}/predictions/${ensemble_method}${alg}_${i}sp_train.npz" \
		--valid-data "superpixel_data/valid_examples_${i}sp.npz" --valid-predictions-output "${results_dir}/predictions/${ensemble_method}${alg}_${i}sp_valid.npz" \
		--test-data "superpixel_data/test_examples_${i}sp.npz" --test-predictions-output "${results_dir}/predictions/${ensemble_method}${alg}_${i}sp_test.npz" \
		--model-file "${results_dir}/models/${ensemble_method}${alg}_${i}sp.npz" \
		--summary-file "${run_dir}/superpixel_report_${start_date}.csv"

	# Encode predictions and get pixel-level results
	# pixel_test_results_file="${results_dir}/test_results/${ensemble_method}${alg}_${i}sp_pixel_level.txt"
	# printf "PIXEL LEVEL RESULTS\n\n" > "${pixel_test_results_file}"
	# printf "===================\n" >> "${pixel_test_results_file}"
	# printf "TRAINING\n" >> "${pixel_test_results_file}"
	# printf "===================\n" >> "${pixel_test_results_file}"

	# echo "Encoding training predictions"
	# python encode_predictions.py \
	# 	-m "superpixel_data/train_superpixels_map_${i}sp.npz" \
	# 	-i "${results_dir}/predictions/${ensemble_method}${alg}_${i}sp_train.npz" \
	# 	-e "kit/data_road/training/divided_data/train/image_2" \
	# 	-o "${results_dir}/prediction_images/${i}sp/train/encoded" \
	# 	-ov "${results_dir}/prediction_images/${i}sp/train/encoded_overlay" >/dev/null 2>&1
	# 	# --generate-overlay \

	# python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/train/encoded/" "kit/data_road/training/divided_data/train" "${run_dir}/pixel_report_${start_date}.csv" "${ensemble_method}${alg}" "${params}" "train" "${i}" >> "${pixel_test_results_file}"

	# printf "\n===================\n" >> "${pixel_test_results_file}"
	# printf "VALID\n" >> "${pixel_test_results_file}"
	# printf "===================\n" >> "${pixel_test_results_file}"

	# echo "Encoding valid predictions"
	# python encode_predictions.py \
	# 	-m "superpixel_data/valid_superpixels_map_${i}sp.npz" \
	# 	-i "${results_dir}/predictions/${ensemble_method}${alg}_${i}sp_valid.npz" \
	# 	-e "kit/data_road/training/divided_data/valid/image_2" \
	# 	-o "${results_dir}/prediction_images/${i}sp/valid/encoded" \
	# 	-ov "${results_dir}/prediction_images/${i}sp/valid/encoded_overlay" >/dev/null 2>&1
	# 	# --generate-overlay \

	# python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/valid/encoded/" "kit/data_road/training/divided_data/valid" "${run_dir}/pixel_report_${start_date}.csv" "${ensemble_method}${alg}" "${params}" "valid" "${i}" >> "${pixel_test_results_file}"

	printf "\n===================\n" > "${pixel_test_results_file}"
	printf "TESTING\n" >> "${pixel_test_results_file}"
	printf "===================\n" >> "${pixel_test_results_file}"

	echo "Encoding test predictions"
	python encode_predictions.py \
		-m "superpixel_data/test_superpixels_map_${i}sp.npz" \
		-i "${results_dir}/predictions/${ensemble_method}${alg}_${i}sp_test.npz" \
		-e "kit/data_road/training/divided_data/test/image_2" \
		-o "${results_dir}/prediction_images/${i}sp/test/encoded" \
		--generate-overlay \
		-ov "${results_dir}/prediction_images/${i}sp/test/encoded_overlay" >/dev/null 2>&1

	python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/test/encoded/" "kit/data_road/training/divided_data/test" "${run_dir}/pixel_report_${start_date}.csv" "${ensemble_method}${alg}" "${params}" "test" "${i}" >> "${pixel_test_results_file}"

	# Write param config to a file
	echo ${params} > "${results_dir}/params_configuration.txt"
done
