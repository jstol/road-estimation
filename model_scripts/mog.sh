#!/usr/bin/env bash
alg="mog"
params_set=( \
	'{"n_components": 1}' \
	'{"n_components": 2}' \
	'{"n_components": 5}' \
	'{"n_components": 10}' \
	'{"n_components": 15}' \
	'{"n_components": 20}' \
	'{"n_components": 25}' \
	'{"n_components": 50}' \
	'{"n_components": 75}' \
	'{"n_components": 100}' \
	'{"n_components": 150}' \
	'{"n_components": 250}' \

)
start_date=$(date +"%m-%d-%Y-%s")

count=1
for params in "${params_set[@]}"
do
	results_subdir="${start_date}/${count}"
	./run_main.sh "${alg}" "${params}" "${results_subdir}" "${start_date}"

	let count++
done

echo "====================================="
echo "DONE"
echo "====================================="
