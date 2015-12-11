#!/usr/bin/env bash
alg="knn"
params_set=( \
	'{"k": 1}' \
	'{"k": 3}' \
	'{"k": 5}' \
	'{"k": 7}' \
	'{"k": 9}' \
	'{"k": 11}' \
	'{"k": 15}' \
	'{"k": 21}' \
	'{"k": 51}' \
	'{"k": 99}' \
	'{"k": 251}' \
	'{"k": 501}' \
	'{"k": 999}' \
#	'{"k": }' \
#	'{"k": 1000}' \
)

for params in "${params_set[@]}"
do
	results_subdir=$(date +"%m-%d-%Y_%s")
	./run_main.sh "${alg}" "${params}" "${results_subdir}"
done

echo "====================================="
echo "DONE"
echo "====================================="
