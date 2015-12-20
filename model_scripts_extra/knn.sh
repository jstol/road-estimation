#!/usr/bin/env bash
alg="knn"
params_set=( \
	'{"k": 1}' \
	'{"k": 3}' \
	'{"k": 7}' \
	'{"k": 15}' \
	'{"k": 51}' \

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
