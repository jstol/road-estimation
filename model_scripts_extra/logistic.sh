#!/usr/bin/env bash
alg="logistic"
params_set=( \
	'{"penalty": "l2", "C": 10}' \
	'{"penalty": "l2", "C": 2}' \
	'{"penalty": "l2", "C": 1}' \

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
