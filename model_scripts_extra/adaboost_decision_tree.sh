#!/usr/bin/env bash
alg="decision_tree"
ensemble_method="adaboost"
params_set=( \
	'{"algorithm_name": "decision_tree", "criterion": "gini", "n_estimators": 15, "max_depth": 7, "min_samples_split": 2}' \
	'{"algorithm_name": "decision_tree", "criterion": "gini", "n_estimators": 15, "max_depth": 10, "min_samples_split": 2}' \
	'{"algorithm_name": "decision_tree", "criterion": "gini", "n_estimators": 25, "max_depth": 7, "min_samples_split": 2}' \
	'{"algorithm_name": "decision_tree", "criterion": "gini", "n_estimators": 25, "max_depth": 10, "min_samples_split": 2}' \
)
start_date=$(date +"%m-%d-%Y-%s")

count=1
for params in "${params_set[@]}"
do
	results_subdir="${start_date}/${count}"
	./run_main_extra.sh "${alg}" "${params}" "${results_subdir}" "${start_date}" "${ensemble_method}"

	let count++
done

echo "====================================="
echo "DONE"
echo "====================================="
