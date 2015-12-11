#!/usr/bin/env bash
alg="knn"
params_set=( \
	'{"k": 1}' \
	'{"k": 10}' \
	'{"k": 100}' \
	'{"k": 1000}' \
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
