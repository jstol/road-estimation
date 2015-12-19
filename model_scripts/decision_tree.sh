#!/usr/bin/env bash
alg="decision_tree"
params_set=( \
	# '{"criterion": "gini", "min_samples_split": 2}' \
	# '{"criterion": "gini", "min_samples_split": 5}' \
	# '{"criterion": "gini", "min_samples_split": 10}' \
	# '{"criterion": "gini", "min_samples_split": 20}' \
	# '{"criterion": "gini", "min_samples_split": 50}' \y
	# '{"criterion": "gini", "min_samples_split": 100}' \
	# '{"criterion": "gini", "max_depth": 2, "min_samples_split": 2}' \
	# '{"criterion": "gini", "max_depth": 2, "min_samples_split": 5}' \
	# '{"criterion": "gini", "max_depth": 2, "min_samples_split": 10}' \
	# '{"criterion": "gini", "max_depth": 2, "min_samples_split": 20}' \
	# '{"criterion": "gini", "max_depth": 2, "min_samples_split": 50}' \
	# '{"criterion": "gini", "max_depth": 2, "min_samples_split": 100}' \
	# '{"criterion": "gini", "max_depth": 5, "min_samples_split": 2}' \
	# '{"criterion": "gini", "max_depth": 5, "min_samples_split": 5}' \
	# '{"criterion": "gini", "max_depth": 5, "min_samples_split": 10}' \
	# '{"criterion": "gini", "max_depth": 5, "min_samples_split": 20}' \
	# '{"criterion": "gini", "max_depth": 5, "min_samples_split": 50}' \
	# '{"criterion": "gini", "max_depth": 5, "min_samples_split": 100}' \
	# '{"criterion": "gini", "max_depth": 7, "min_samples_split": 2}' \
	# '{"criterion": "gini", "max_depth": 7, "min_samples_split": 5}' \
	# '{"criterion": "gini", "max_depth": 7, "min_samples_split": 10}' \
	# '{"criterion": "gini", "max_depth": 7, "min_samples_split": 20}' \
	# '{"criterion": "gini", "max_depth": 7, "min_samples_split": 50}' \
	# '{"criterion": "gini", "max_depth": 7, "min_samples_split": 100}' \
	# '{"criterion": "gini", "max_depth": 10, "min_samples_split": 2}' \
	# '{"criterion": "gini", "max_depth": 10, "min_samples_split": 5}' \
	# '{"criterion": "gini", "max_depth": 10, "min_samples_split": 10}' \
	# '{"criterion": "gini", "max_depth": 10, "min_samples_split": 20}' \
	# '{"criterion": "gini", "max_depth": 10, "min_samples_split": 50}' \
	# '{"criterion": "gini", "max_depth": 10, "min_samples_split": 100}' \
	# '{"criterion": "entropy", "min_samples_split": 2}' \
	# '{"criterion": "entropy", "min_samples_split": 5}' \
	# '{"criterion": "entropy", "min_samples_split": 10}' \
	# '{"criterion": "entropy", "min_samples_split": 20}' \
	# '{"criterion": "entropy", "min_samples_split": 50}' \
	# '{"criterion": "entropy", "min_samples_split": 100}' \
	'{"criterion": "entropy", "max_depth": 2, "min_samples_split": 2}' \
	'{"criterion": "entropy", "max_depth": 2, "min_samples_split": 5}' \
	'{"criterion": "entropy", "max_depth": 2, "min_samples_split": 10}' \
	'{"criterion": "entropy", "max_depth": 2, "min_samples_split": 20}' \
	'{"criterion": "entropy", "max_depth": 2, "min_samples_split": 50}' \
	'{"criterion": "entropy", "max_depth": 2, "min_samples_split": 100}' \
	'{"criterion": "entropy", "max_depth": 5, "min_samples_split": 2}' \
	'{"criterion": "entropy", "max_depth": 5, "min_samples_split": 5}' \
	'{"criterion": "entropy", "max_depth": 5, "min_samples_split": 10}' \
	'{"criterion": "entropy", "max_depth": 5, "min_samples_split": 20}' \
	'{"criterion": "entropy", "max_depth": 5, "min_samples_split": 50}' \
	'{"criterion": "entropy", "max_depth": 5, "min_samples_split": 100}' \
	'{"criterion": "entropy", "max_depth": 7, "min_samples_split": 2}' \
	'{"criterion": "entropy", "max_depth": 7, "min_samples_split": 5}' \
	'{"criterion": "entropy", "max_depth": 7, "min_samples_split": 10}' \
	'{"criterion": "entropy", "max_depth": 7, "min_samples_split": 20}' \
	'{"criterion": "entropy", "max_depth": 7, "min_samples_split": 50}' \
	'{"criterion": "entropy", "max_depth": 7, "min_samples_split": 100}' \
	'{"criterion": "entropy", "max_depth": 10, "min_samples_split": 2}' \
	'{"criterion": "entropy", "max_depth": 10, "min_samples_split": 5}' \
	'{"criterion": "entropy", "max_depth": 10, "min_samples_split": 10}' \
	'{"criterion": "entropy", "max_depth": 10, "min_samples_split": 20}' \
	'{"criterion": "entropy", "max_depth": 10, "min_samples_split": 50}' \
	'{"criterion": "entropy", "max_depth": 10, "min_samples_split": 100}' \
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
