#!/usr/bin/env bash
alg="logistic"
params_set=( \
	'{"penalty": "l1", "C": 200000}' \
	'{"penalty": "l1", "C": 20000}' \
	'{"penalty": "l1", "C": 2000}' \
	'{"penalty": "l1", "C": 200}' \
	'{"penalty": "l1", "C": 100}' \
	'{"penalty": "l1", "C": 20}' \
	'{"penalty": "l1", "C": 10}' \
	'{"penalty": "l1", "C": 2}' \
	'{"penalty": "l1", "C": 1}' \
	'{"penalty": "l1", "C": 0.5}' \
	'{"penalty": "l1", "C": 0.2}' \
	'{"penalty": "l1", "C": 0.1}' \
	'{"penalty": "l1", "C": 0.05}' \
	'{"penalty": "l1", "C": 0.02}' \
	'{"penalty": "l2", "C": 200000}' \
	'{"penalty": "l2", "C": 20000}' \
	'{"penalty": "l2", "C": 2000}' \
	'{"penalty": "l2", "C": 200}' \
	'{"penalty": "l2", "C": 100}' \
	'{"penalty": "l2", "C": 20}' \
	'{"penalty": "l2", "C": 10}' \
	'{"penalty": "l2", "C": 2}' \
	'{"penalty": "l2", "C": 1}' \
	'{"penalty": "l2", "C": 0.5}' \
	'{"penalty": "l2", "C": 0.2}' \
	'{"penalty": "l2", "C": 0.1}' \
	'{"penalty": "l2", "C": 0.05}' \
	'{"penalty": "l2", "C": 0.02}' \
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
