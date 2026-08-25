#!/usr/bin/bash

results_dir="./results_openai"
data_dir="/data1/toghrul/datasets" # "./data/eval_res/eval_ds_2" # 
model_name_1="gpt-4o"
model_name_2="gpt-5.2"
model_name_3="o4-mini"
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

# if [$2 -lt 21] : then
#     python main.py --model-name $model_name --dataset-name $dataset_name --device cuda:0 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 600 $
# else
#     echo "pick higher"
# fi

# python main.py --model-name $model_name_1 --dataset-name $ds_name1 --num-in-context-samples 5 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --baseline-methods seq_likelihood cot_seq_likelihood cot_qual_verbalized_uncertainty ps_seq_likelihood ts_seq_likelihood lmvslm
# python main.py --model-name $model_name_2 --dataset-name $ds_name1 --num-in-context-samples 5 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --knock
# python main.py --model-name $model_name_1 --dataset-name $ds_name2 --num-in-context-samples 5 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --baseline-methods seq_likelihood cot_seq_likelihood cot_qual_verbalized_uncertainty ps_seq_likelihood ts_seq_likelihood lmvslm
# python main.py --model-name $model_name_2 --dataset-name $ds_name2 --num-in-context-samples 5 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --knock
python main.py --model-name $model_name_2 --dataset-name $ds_name1 --num-in-context-samples 10 --data-dir $data_dir --results-dir $results_dir --calibration-num-steps 200 --baseline-methods cot_qual_verbalized_uncertainty qual_verbalized_uncertainty
