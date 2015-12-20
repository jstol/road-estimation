#!/usr/bin/env bash
alg="decision_tree"
params_set=( \
	'{"criterion": "gini", "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "max_depth": 20, "min_samples_split": 2}' \
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
