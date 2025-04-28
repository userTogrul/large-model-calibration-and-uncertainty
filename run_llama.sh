#!/usr/bin/bash

results_dir="./results_llama"
data_dir="./data"
model_name="meta-llama/Llama-3.1-8B-Instruct"
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

# if [$2 -lt 21] : then
#     python main.py --model-name $model_name --dataset-name $dataset_name --device cuda:0 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 600 $
# else
#     echo "pick higher"
# fi

python main.py --model-name $model_name --dataset-name $dataset_name --device cuda:0 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --knock
python main.py --model-name $model_name --dataset-name $ds_name2 --device cuda:0 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --knock

