#!/usr/bin/env bash
alg="mog"
params_set=( \
	'{"n_components": 1}' \
	'{"n_components": 25}' \
	'{"n_components": 100}' \
	'{"n_components": 150}' \

)
start_date=$(date +"%m-%d-%Y-%s")

count=1
for params in "${params_set[@]}"
do
	results_subdir="${start_date}/${count}"
	./run_main_extra.sh "${alg}" "${params}" "${results_subdir}" "${start_date}"

	let count++
done

echo "====================================="
echo "DONE"
echo "====================================="
