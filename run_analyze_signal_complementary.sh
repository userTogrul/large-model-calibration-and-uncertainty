#!/usr/bin/bash

data_dir="/data1/toghrul/datasets" #
model_name_1="gpt-4o"
model_name_2="gpt-5.2"
model_name_3="o4-mini"
ds_name1="trivia_qa"
ds_name2="truthful_qa"
ds_name3="natural_questions"

echo "Run $1" 
python analyze_signal_complementarity.py --model-name $model_name_1 --dataset-name $ds_name2 --max-examples 100 --n-consistency-samples 5 --data-dir $data_dir --output "results/complementarity_truthfulqa_gpt4o.json" 
# python code/analyze_signal_complementarity.py --model-name $model_name_2 --dataset-name $ds_name2 --max-examples 100 --n-consistency-samples 5 --data-dir $data_dir --output "results/complementarity_truthfulqa_gpt5.2.json"
# python code/analyze_signal_complementarity.py --model-name $model_name_3 --dataset-name $ds_name2 --max-examples 100 --n-consistency-samples 5 --data-dir $data_dir --output "results/complementarity_truthfulqa_o4-mini.json"
