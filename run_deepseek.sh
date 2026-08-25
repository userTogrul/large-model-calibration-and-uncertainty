#!/usr/bin/bash

results_dir="./results/results_deepseek_1"
data_dir="/data1/toghrul/datasets" #
model_name_1="aiproxy/deepseek-reasoner" # Deepseek-R1
ds_name1="trivia_qa"
ds_name2="truthful_qa"
ds_name3="natural_questions"
echo "Run $1"

while true; do
    read -p "Do you want to run QA task?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no." ;;
    esac
done

# python main.py --model-name $model_name_1 --dataset-name $dataset_name --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --baseline-methods cot_qual_verbalized_uncertainty qual_verbalized_uncertainty lmvslm hallumeasure
# python main.py --model-name $model_name_1 --dataset-name $ds_name2 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --baseline-methods cot_qual_verbalized_uncertainty qual_verbalized_uncertainty lmvslm hallumeasure
python main.py --model-name $model_name_1 --dataset-name $ds_name2 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --baseline-methods ourmethod
