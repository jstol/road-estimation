#!/usr/bin/env bash
alg="extratrees"
params_set=( \
	'{"criterion": "gini", "n_estimators": 5, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 5, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 5, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 5, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 5, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 5, "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 10, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 10, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 10, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 10, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 10, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 10, "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 15, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 15, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 15, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 15, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 15, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 15, "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 25, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 25, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 25, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 25, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 25, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 25, "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 50, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 50, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 50, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 50, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 50, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 50, "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 100, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 100, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 100, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 100, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 100, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 100, "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 150, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 150, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 150, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 150, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 150, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 150, "max_depth": 15, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 200, "max_depth": 1, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 200, "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 200, "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 200, "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 200, "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "gini", "n_estimators": 200, "max_depth": 15, "min_samples_split": 2}' \
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
