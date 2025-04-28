#!/usr/bin/bash

results_dir="./results/results_deepseek_1"
data_dir="./data"
model_name_1="deepseek-reasoner"
dataset_name="trivia_qa"
ds_name2="truthful_qa"

echo "Run $1"

while true; do
    read -p "Do you want to run QA task?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no." ;;
    esac
done

python main.py --model-name $model_name_1 --dataset-name $dataset_name --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --knock
python main.py --model-name $model_name_1 --dataset-name $ds_name2 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --knock
