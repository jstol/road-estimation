#!/usr/bin/env bash
alg="neural_net"
params='{"list_of_layers_params": [{"role":"hidden", "type":"Tanh", "units": 50, "weight_decay": 0.0000001}, {"role":"hidden", "type":"Tanh", "units": 25, "weight_decay": 0.0000001}, {"role":"hidden", "type":"Tanh", "units": 5, "weight_decay": 0.0000001}, {"role":"output", "type":"Softmax"}]}'
param_configuration_subdir="final"
start_date=$(date +"%m-%d-%Y-%s")

i=5000 # number of superpixels
run_dir="final_test_results/${alg}"
results_dir="${run_dir}/${param_configuration_subdir}"
mkdir -p results_dir

echo "====================================="
echo "Running for ${alg} - ${params}"
echo "====================================="
# Make prediction
python predict_final.py

# Encode predictions and get pixel-level results
pixel_test_results_file="${results_dir}/test_results/${alg}_${i}sp_pixel_level.txt"
printf "PIXEL LEVEL RESULTS\n\n" > "${pixel_test_results_file}"
printf "===================\n" >> "${pixel_test_results_file}"
printf "ON TEST SET\n" >> "${pixel_test_results_file}"
printf "===================\n" >> "${pixel_test_results_file}"

echo "Encoding test predictions"
python encode_predictions.py \
	-m "superpixel_data/test_superpixels_map_${i}sp.npz" \
	-i "${results_dir}/predictions/${alg}_${i}sp_test.npz" \
	-e "kit/data_road/training/divided_data/test/image_2" \
	-o "${results_dir}/prediction_images/${i}sp/test/encoded" \
	--generate-overlay \
	-ov "${results_dir}/prediction_images/${i}sp/test/encoded_overlay" >/dev/null 2>&1

python kit/devkit_road/python/evaluateRoad.py  "${results_dir}/prediction_images/${i}sp/test/encoded/" "kit/data_road/training/divided_data/test" "${run_dir}/pixel_report_${start_date}.csv" "${alg}" "${params}" "test" "${i}" >> "${pixel_test_results_file}"

# Write param config to a file
echo ${params} > "${results_dir}/params_configuration.txt"
