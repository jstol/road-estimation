#!/usr/bin/env bash
alg="neural_net"
params_set=( \
	'{"list_of_layers_params": [{"role":"hidden", "type":"Tanh", "units": 50, "weight_decay": 0.0000001}, {"role":"hidden", "type":"Tanh", "units": 25, "weight_decay": 0.0000001}, {"role":"output", "type":"Softmax"}]}' \
)

start_date=$(date +"%m-%d-%Y-%s")

count=1
for params in "${params_set[@]}"
do
	results_subdir="${start_date}/${count}"
	./run_main_extra_final.sh "${alg}" "${params}" "${results_subdir}" "${start_date}"

	let count++
done

echo "====================================="
echo "DONE"
echo "====================================="
