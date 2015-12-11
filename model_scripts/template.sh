#!/usr/bin/env bash
alg="ALGORITHM"
params_set=( \
	'{"p": 1}' \
	'{"p": 10}' \
	'{"p": 100}'
)

for params in "${params_set[@]}"
do
	results_subdir=$(date +"%m-%d-%Y_%s")
	./run_main.sh "${alg}" "${params}" "${results_subdir}"
done

echo "====================================="
echo "DONE"
echo "====================================="
