#!/usr/bin/env bash
alg=$1
params=$2
param_configuration_subdir=$3
start_date=$4
ensemble_method=$5

i=5000 # number of superpixels
run_dir="final_test_results/${ensemble_method}${alg}"
results_dir="${run_dir}/${param_configuration_subdir}"
mkdir -p results_dir

echo "====================================="
echo "Running for ${ensemble_method} ${alg} - ${params}"
echo "====================================="
# Make prediction
ptyhon predict_final.py

# Encode predictions and get pixel-level results
pixel_test_results_file="${results_dir}/test_results/${ensemble_method}${alg}_${i}sp_pixel_level.txt"
printf "PIXEL LEVEL RESULTS\n\n" > "${pixel_test_results_file}"
printf "===================\n" >> "${pixel_test_results_file}"
printf "ON TEST SET\n" >> "${pixel_test_results_file}"
printf "===================\n" >> "${pixel_test_results_file}"

echo "Encoding test predictions"
python encode_predictions.py \
	-m "superpixel_data/test_superpixels_map_${i}sp.npz" \
	-i "${results_dir}/predictions/${ensemble_method}${alg}_${i}sp_test.npz" \
	-e "kit/data_road/testing/divided_data/test/image_2" \
	-o "${results_dir}/prediction_images/${i}sp/test/encoded" \
	--generate-overlay \
	-ov "${results_dir}/prediction_images/${i}sp/test/encoded_overlay" >/dev/null 2>&1

python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/test/encoded/" "kit/data_road/testing/divided_data/test" "${run_dir}/pixel_report_${start_date}.csv" "${ensemble_method}${alg}" "${params}" "test" "${i}" >> "${pixel_test_results_file}"

# Write param config to a file
echo ${params} > "${results_dir}/params_configuration.txt"
