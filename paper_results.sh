#!/bin/bash


for DATASET in "airlines" "electricity" "kdd99" "wisdm" "covtype" "nomao" "agr_a" "agr_g" "rbf_f" "rbf_m" "led_a" "led_g"
do
	for SEED in 40 41 42 43 44
	do
	  echo "Run experiments with dataset $DATASET and seed $SEED"
    python main.py -d "$DATASET" -s "$SEED"
	done
done