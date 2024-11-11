#!/usr/bin/bash

results_dir="./results_mistral"
data_dir="/data1/toghrul/datasets/"
model_name="mistralai/Mistral-7B-Instruct-v0.3"
dataset_name="trivia_qa"

echo "Run $1"

while true; do
    read -p "Do you want to run QA task?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no." ;;
    esac
done

# if [$2 -lt 21] : then
#     python main.py --model-name $model_name --dataset-name $dataset_name --device cuda:0 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 600 $
# else
#     echo "pick higher"
# fi

python main.py --model-name $model_name --dataset-name $dataset_name --device cuda:1 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 600 --knock

