#!/usr/bin/env bash
alg="logistic"
params_set=( \
	'{"penalty": "l1", "regularization_term": 0.005}' \
	'{"penalty": "l1", "regularization_term": 0.01}' \
	'{"penalty": "l1", "regularization_term": 0.05}' \
	'{"penalty": "l1", "regularization_term": 0.1}' \
	'{"penalty": "l1", "regularization_term": 0.5}' \
	'{"penalty": "l1", "regularization_term": 1}' \
	'{"penalty": "l1", "regularization_term": 2}' \
	'{"penalty": "l1", "regularization_term": 5}' \
	'{"penalty": "l1", "regularization_term": 10}' \
	'{"penalty": "l1", "regularization_term": 20}' \
	'{"penalty": "l1", "regularization_term": 50}' \
	'{"penalty": "l2", "regularization_term": 0.005}' \
	'{"penalty": "l2", "regularization_term": 0.01}' \
	'{"penalty": "l2", "regularization_term": 0.05}' \
	'{"penalty": "l2", "regularization_term": 0.1}' \
	'{"penalty": "l2", "regularization_term": 0.5}' \
	'{"penalty": "l2", "regularization_term": 1}' \
	'{"penalty": "l2", "regularization_term": 2}' \
	'{"penalty": "l2", "regularization_term": 5}' \
	'{"penalty": "l2", "regularization_term": 10}' \
	'{"penalty": "l2", "regularization_term": 20}' \
	'{"penalty": "l2", "regularization_term": 50}' \
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
