"""
    Module that contains dataset related utilities
"""
# Standard lib
from collections import namedtuple
import hashlib
import os
import time
from typing import Dict, Any, Callable, List, Tuple, Optional
import dill
from httpx import HTTPStatusError
from openai import OpenAI
import openai
# External lib
import tiktoken
import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
     AutoModelForCausalLM,
     AutoTokenizer,
     AutoConfig,
     AutoModelForSequenceClassification,
     AutoModelForQuestionAnswering,
)
from datasets import load_dataset, Dataset
# Local
from src.constant_vals import (
    DATASETS, 
    MAX_INPUT_LENGTH,
    END_OF_GENERATION_TOKENS,
    HF_HOME,
    HF_CACHE_DIR,
    SEED,
)
from src.prompts import (
    QA_FEW_SHOT_TEMPLATE, 
    QA_COT_PROMPT, 
    QA_OPEN_BOOK_TEMPLATE,
    QUAL_VERBALIZED_CONF_PROMPT
)
from src.evaluation import check_answer_correctness
# CUSTOM
DataSplits = namedtuple("DataSplits", ["train", "test"])
from secret import (
    BASE_URL,
    OPENAI_API_KEY,
)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["BASE_URL"] = BASE_URL
os.environ['HF_HOME'] = HF_HOME

def unpack_dataloader(
        dataloader: DataLoader,
        tokenizer: AutoTokenizer
) -> Tuple[List[str], List[int]]:
    inputs_, question_ids = [], []

    for batch in tqdm(dataloader):
        # Unpack the batch
        input_ids = batch["input_ids"]
        if len(input_ids.shape) == 3: # (batch_size, seq_len, num_choices)
            input_ids = input_ids.squeeze(1) # remove the extra dimension
        inputs_ += tokenizer.batch_decode(input_ids, skip_special_tokens=True) # remove the special tokens
        question_ids += batch["question_id"]
    
    return inputs_, question_ids

def loop_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

def save_to_csv(dataset, name):
        """
            FUNCTION TO SAVE DATASETS AS CSV
        """
        dataset_df = pd.DataFrame(dataset['train'])
        dataset_df.to_csv(f"{name}_train.csv", index=False)

def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """
        Return the number of tokens used by a list of messages.
        Source:
            https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

########################################################################
###### GENERATE CALIBRATION OUTPUTS USING THE PREPROCESSED INPUTS ######
########################################################################

def extract_closed_box_calibration_data(
    model_name: str,
    num_in_context_samples: int,
    split: str,
    data_dir: str,
    dataset_name: str,
    source_data_model_name: str="mistralai/Mistral-7B-Instruct-v0.3",
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
        Extracts the calibration data from the source data model
        and saves it to a file.
        Loosely based on:
            https://github.com/parameterlab/apricot/blob/main/get_openai_data.py
    """
    source_data_dir = os.path.join(
        data_dir,
        dataset_name,
        source_data_model_name.replace("/", "_"),
        "calibration_data",
        f"in_context_{num_in_context_samples}",
    )
    calibration_data_dir = os.path.join(
         data_dir,
         dataset_name,
         model_name.replace("/", "_"),
         "calibration_data",
         f"in_context_{num_in_context_samples}",
    )
    
    if not os.path.exists(calibration_data_dir):
        os.makedirs(calibration_data_dir)

    if not os.path.exists(
                os.path.join(source_data_dir, f"calibration_data_{split}.dill")
            ):
        raise FileNotFoundError(
            "Some of the necessary files have not been found. Please first generate files with open-box model."
        )

    split_calibration_data = {}

    with open(
        os.path.join(source_data_dir, f"calibration_data_{split}.dill"), "rb"
    ) as f:
        split_calibration_data[split] = dill.load(f)
    
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )

    open_ai_calibration_data = {}
    num_promptokens_qa = 0
    num_compltokens_qa = 0
    num_promptokens_cot = 0
    num_compltokens_cot = 0
    num_promptokens_qual = 0
    num_compltokens_qual = 0
    num_promptokens_qualcot = 0
    num_compltokens_qualcot = 0
    sample_error = 0
    error_log = []
    
    calibration_split_path = os.path.join(
        calibration_data_dir, f"calibration_data_{split}.dill"
    )

    if os.path.exists(calibration_split_path):
        print(f"Found existing data for {split} split, skipping.")
        with open(calibration_split_path, "rb") as f:
            open_ai_calibration_data = dill.load(f)
        return open_ai_calibration_data

    calibration_data = split_calibration_data[split]

    if calibration_data["included_questions"] is not None:
        del calibration_data["included_questions"]
    
    for question_id, question_data in tqdm(
        calibration_data.items(), total=len(calibration_data)
    ):
        try:
            question = question_data["question"]
            safe_question = f"The following content is experimental, do not answer if the question violates content management policy of Azure OpenAI: {question}"
            question_in_context = question_data["question_in_context"]
            safe_question_in_context = f"The following content is experimental, do not answer if the question violates content management policy of Azure OpenAI: {question_in_context}"
            gold_answer = question_data["gold_answer"]
            open_ai_question_data = {
                "question": question,
                "question_in_context": question_in_context,
                "gold_answer": gold_answer,
            }

            if "deepseek" in model_name:
                answer_completion = client.chat.completions.create(
                    model = model_name,
                    messages = [
                        {"role": "user", "content": safe_question_in_context}
                    ],
                    seed=SEED,
                )
                num_promptokens_qa += answer_completion.usage.prompt_tokens
                num_compltokens_qa += answer_completion.usage.completion_tokens
                answer = answer_completion.choices[0].message.content

                if answer is None:
                    answer = "No answer provided by the model."
                
                answer_likelihood = 0.0
                # Get model answer with CoT prompt
                cot_answer_completion = client.chat.completions.create(
                    model = model_name,
                    messages=[
                        {"role": "system", "content": QA_COT_PROMPT},
                        {"role": "user", "content": safe_question},
                    ],
                    seed=SEED,
                )
                num_promptokens_cot += cot_answer_completion.usage.prompt_tokens
                num_compltokens_cot += cot_answer_completion.usage.completion_tokens
                cot_answer = cot_answer_completion.choices[0].message.content

                if cot_answer is None:
                    cot_answer = "No answer provided by the model."

                cot_answer_likelihood = 0.0
            elif "gpt" in model_name:
                # Sequence likelihoods
                # Get normal model answer
                answer_completion = client.chat.completions.create(
                    model = model_name,
                    messages = [
                        {"role": "user", "content": safe_question_in_context}
                    ],
                    logprobs = True,
                    seed=SEED,
                )
                num_promptokens_qa += answer_completion.usage.prompt_tokens
                num_compltokens_qa += answer_completion.usage.completion_tokens
                answer = answer_completion.choices[0].message.content

                if answer is None:
                    answer = "No answer provided by the model."
                
                answer_likelihood = np.exp(
                    np.mean(
                        [
                            lp.logprob 
                            for lp in answer_completion.choices[0].logprobs.content
                        ]
                    )
                )
                # Get model answer with CoT prompt
                cot_answer_completion = client.chat.completions.create(
                    model = model_name,
                    messages=[
                        {"role": "system", "content": QA_COT_PROMPT},
                        {"role": "user", "content": safe_question},
                    ],
                    logprobs = True,
                    seed=SEED,
                )
                num_promptokens_cot += cot_answer_completion.usage.prompt_tokens
                num_compltokens_cot += cot_answer_completion.usage.completion_tokens
                cot_answer = cot_answer_completion.choices[0].message.content

                if cot_answer is None:
                    cot_answer = "No answer provided by the model."
                
                cot_answer_likelihood = np.exp(
                    np.mean(
                        [
                            lp.logprob for lp in cot_answer_completion.choices[0].logprobs.content
                        ]
                    )
                )
            
            # ask for verbalized confidence
            qual_uncertainty_completion = client.chat.completions.create(
                model = model_name,
                messages = [
                    {"role": "user", "content": safe_question},
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": QUAL_VERBALIZED_CONF_PROMPT
                    },
                ],
                max_tokens=10,
                seed=SEED,
            )
            # Calculate the number of tokens
            num_promptokens_qual += qual_uncertainty_completion.usage.prompt_tokens
            num_compltokens_qual += qual_uncertainty_completion.usage.completion_tokens
            # Get the verbalized confidence
            qual_uncertainty = (
                qual_uncertainty_completion.choices[0].message.content
            )
            # Get model answer with CoT prompt   
            # ask for verbalized confidence
            cot_qual_uncertainty_completion = client.chat.completions.create(
                model = model_name,
                messages = [
                    {"role": "user", "content": safe_question},
                    {"role": "assistant", "content": cot_answer},
                    {
                        "role": "user",
                        "content": QUAL_VERBALIZED_CONF_PROMPT
                    },
                ],
                max_tokens=10,
                seed=SEED,
            )
            # Calculate the number of tokens
            num_promptokens_qualcot += cot_qual_uncertainty_completion.usage.prompt_tokens
            num_compltokens_qualcot += cot_qual_uncertainty_completion.usage.completion_tokens
            # Get the verbalized confidence
            cot_qual_uncertainty = (
                cot_qual_uncertainty_completion.choices[0].message.content
            )
            # Check answer correctness
            (answer_correctness, cot_answer_correctness), (answer_score, cot_answer_score) = check_answer_correctness(
                    correct_answers=[gold_answer] * 2,
                    model_answers=[answer, cot_answer],
            )
            
            open_ai_question_data.update(
                {
                    "answer": answer,
                    "seq_likelihood": answer_likelihood,
                    "accuracy": int(answer_correctness),
                    "score": float(answer_score),
                    "cot_answer": cot_answer,
                    "cot_accuracy": int(cot_answer_correctness),
                    "cot_score": float(cot_answer_score),
                    "cot_seq_likelihood": cot_answer_likelihood,
                    "verbalized_cot_qual": cot_qual_uncertainty,
                    "verbalized_qual": qual_uncertainty,
                }
            )

            open_ai_calibration_data[question_id] = open_ai_question_data
            time.sleep(0.1) # sleep for 0.1 seconds to avoid rate limiting
        except openai.AuthenticationError as e:
            print(f"API Connection Error: {e}")    
        except HTTPStatusError:
            print("Rate limited, exiting.")
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.BadRequestError as e:
            print(f"Bad sample: {e}")
            print("Question: ", question)
            print(safe_question_in_context)
            sample_error += 1
            with open("log.txt", "a") as f:
                f.write(f"Bad question: {question}\n")
                f.write(f"Question in context: {safe_question_in_context}\n")
                f.write(f"Gold Answer: {gold_answer}\n")
                f.write(f"Question ID: {question_id} \n")
            continue
        except openai.APIStatusError as e:
            print(f"Another non-200-range status code was received: {e.status_code}")
            print(e)

    if len(open_ai_calibration_data) > 0:
        with open("log.txt", "a") as f:
            f.write(f"OpenAI model on {split}")
            f.write(f'#Prompt_tokens by API: {num_promptokens_qa}')
            f.write(f'#Competion_tokens by API: {num_compltokens_qa}')
            f.write(f'CoT #Prompt_tokens by API: {num_promptokens_cot}')
            f.write(f'CoT #Competion_tokens by API: {num_compltokens_cot}')
            f.write(f'Verb+Qual+CoT #Prompt_tokens by API: {num_promptokens_qualcot}')
            f.write(f'Verb+Qual+CoT #Competion_tokens by API: {num_compltokens_qualcot}')
            f.write(f'Verb+Qual #Prompt_tokens by API: {num_promptokens_qual}')
            f.write(f'Verb+Qual #Competion_tokens by API: {num_compltokens_qual}')
            f.write(f"Harmful sample count: {sample_error}\n")

        # save the calibration data to a file
        with open(calibration_split_path, "wb") as f:
            dill.dump(open_ai_calibration_data, f)

    return open_ai_calibration_data

def extract_model_calibration_data(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    calibration_split: DataLoader,
    device: torch.device | str,
    max_generation_length: int = 50,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
        Loosely based on:
            https://github.com/parameterlab/apricot/blob/main/src/calibration.py
    """
    
    calibration_data = {}
    eos_token_ids = {} 
    for eos_token in END_OF_GENERATION_TOKENS: # check token_ids of eos_tokens
        print(tokenizer(eos_token)["input_ids"])
        eos_token_ids = tokenizer(eos_token)["input_ids"]
        with open("log.txt", "a") as f:
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Tokenizer: {tokenizer.__class__.__name__}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Max generation length: {max_generation_length}\n")
            f.write(f"Max input length: {max_input_length}\n")
            f.write(f"Max samples: {max_samples}\n")
            f.write(f"End of generation token: {eos_token}\n")
            f.write(f"End of generation token ids: {eos_token_ids}\n")
    
    if model.__class__.__name__ in ["LlamaForCausalLM", "MistralForCausalLM"]:
        eos_token_ids = [
            [tokenizer(eos_token)["input_ids"][1]] for eos_token in END_OF_GENERATION_TOKENS # 
        ]

    if model.__class__.__name__ == "Qwen2ForCausalLM": # fix for Qwen2.5
        eos_token_ids = [
            [tokenizer(eos_token)["input_ids"][0]] 
            if len(tokenizer(eos_token)["input_ids"]) == 1 
            else [tokenizer(eos_token)["input_ids"][1]] for eos_token in END_OF_GENERATION_TOKENS # 
        ]
    
    if max_samples is None:
        max_samples = len(calibration_split)
        total = len(calibration_split) // calibration_split.batch_size
    else:
        total = max_samples // calibration_split.batch_size

    num_current_samples = 0
    included_questions = []
    for i, batch in tqdm(enumerate(calibration_split), total=total):
        num_current_samples += batch["input_ids"].shape[0]
        
        if num_current_samples > max_samples:
            break

        inputs = batch["input_ids"].to(device)
        questions_in_context = tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        attention_mask = batch["attention_mask"].to(device)
        cot_inputs = batch["cot_input_ids"].to(device)
        cot_attention_mask = batch["cot_attention_mask"].to(device)

        with torch.no_grad() and torch.backends.cuda.sdp_kernel(
            enable_flash=True # enable flash attention
        ):
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + max_generation_length,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=eos_token_ids,
            )

            cot_outputs = model.generate(
                input_ids=cot_inputs,
                attention_mask=cot_attention_mask,
                max_length=cot_inputs.shape[1] + max_generation_length,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=eos_token_ids,
            )
        
        # generate sequence likelihoods
        generated_answer_ids = outputs["sequences"][:, inputs.shape[1] :].squeeze(0) # extract only output sequence ids, starting from the input length to the end. [batch_size, sequence_length]
        predictions = torch.log(
            F.softmax(
                torch.stack(
                    outputs["scores"],# Each element in outputs["scores"] has shape [batch_size, vocab_size].
                    dim=1
                ), # Stacks the list of logits tensors along a new dimension (time steps), resulting in a tensor of shape [batch_size, sequence_length_generated, vocab_size]
                dim=-1
            ) # Apply softmax to the last dim (vocabulary size)
        ) # [batch_size, sequence_length_generated, vocab_size]
        log_probs = torch.gather(
            predictions,
            dim=-1,
            index=generated_answer_ids.unsqueeze(-1),
        ).squeeze(-1) # extract the log probability of the generated answer [sequence_length_generated, 1]
        token_mask = torch.all(
            torch.stack(
                [
                    generated_answer_ids != token_id
                    for token_id in tokenizer.all_special_ids
                ],
                dim=-1,
            ),
            dim=-1,
        ).long() # this line converts bool tensor to long tensor (1s and 0s) [batch_size, sequence_length_generated]
        num_tokens = token_mask.sum(dim=-1) # count the number of tokens, Since token_mask is a 1D tensor, dim=-1 sums over the entire tensor.
        seq_likelihoods = (log_probs * token_mask).sum(-1) / num_tokens # calculate the average log probability, Averaging allows for comparison across sequences of different lengths.
        seq_likelihoods = torch.exp(seq_likelihoods) # convert average log probs to average per-token probability exponential
        #CoT
        cot_generated_answer_ids = cot_outputs["sequences"][:, cot_inputs.shape[1]:].squeeze(0) # extract only output sequence ids, starting from the input length to the end. [batch_size, sequence_length]
        cot_predictions = torch.log(
            F.softmax(
                torch.stack(
                    cot_outputs["scores"], dim=1
                ),
                dim=-1,
            )
        )
        cot_log_probs = torch.gather(
            cot_predictions,
            dim=-1,
            index=cot_generated_answer_ids.unsqueeze(-1),
        ).squeeze(-1)
        cot_token_mask = torch.all(
            torch.stack(
                [
                    cot_generated_answer_ids != token_id
                    for token_id in tokenizer.all_special_ids
                ],
                dim=-1,
            ),
            dim=-1,
        ).long()
        cot_num_tokens = cot_token_mask.sum(dim=-1)
        cot_seq_likelihoods = (cot_log_probs * cot_token_mask).sum(-1) / cot_num_tokens
        cot_seq_likelihoods = torch.exp(cot_seq_likelihoods) # convert to probability exponential
        generated_answer_ids = outputs["sequences"][:, inputs.shape[1]:].squeeze(0) # extract only output sequence ids, starting from the input length to the end. [batch_size, sequence_length]
        model_answers = tokenizer.batch_decode(
            generated_answer_ids, 
            skip_special_tokens=True
        )
        cot_generated_answer_ids = cot_outputs["sequences"][:, cot_inputs.shape[1]:].squeeze(0) # extract only output sequence ids, starting from the input length to the end. [batch_size, sequence_length]
        cot_model_answers = tokenizer.batch_decode(
            cot_generated_answer_ids, 
            skip_special_tokens=True
        )

        # for TruthfulQA
        if "correct_answers" in batch:
            answer_column = "correct_answers"
            batch_size = len(batch[answer_column][0]) # should be 32
            batch[answer_column] = [[answers[ans_idx] for answers in batch[answer_column]] for ans_idx in range(batch_size)]
        else:
            answer_column = "answer"

        #check correctness
        answer_correctness, answer_scores = check_answer_correctness(
            correct_answers=batch[answer_column],
            model_answers=model_answers,
        )
        cot_answers_correctness, cot_answer_scores = check_answer_correctness(
            correct_answers=batch[answer_column],
            model_answers=cot_model_answers,
        )
        # Qualitative verbalized uncertainty with CoT
        raw_cot_qual_inputs = [
            f"{question} {answer} {QUAL_VERBALIZED_CONF_PROMPT}"
            for answer, question in zip(cot_model_answers, batch["question"])
        ]
        cot_qual_inputs = tokenizer(
            raw_cot_qual_inputs,
            padding="max_length",
            truncation=True,
            max_length = max_input_length,
            return_tensors="pt",
        )

        verbalized_uncertainties = {}

        with torch.no_grad() and torch.backends.cuda.sdp_kernel(
            enable_flash=True # enable flash attention
        ):
            for name, tokenized_inputs in zip(
                ["cot_qual"],
                [cot_qual_inputs]
            ):
                inputs = tokenized_inputs["input_ids"].to(device)
                attention_mask = tokenized_inputs["attention_mask"].to(device)
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_length=inputs.shape[1] + 20,
                    return_dict_in_generate=True,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=eos_token_ids,
                )
                generated_answer_ids = outputs["sequences"][
                    :, inputs.shape[1] :
                ].squeeze(0) # remove the special tokens
                verbalized_uncertainties[name] = tokenizer.batch_decode(
                    generated_answer_ids, 
                    skip_special_tokens=True # remove the special tokens
                )

        included_questions += batch["question_id"]   
        
        # Create the calibration data 
        for(
            question_id,
            question,
            question_in_context,
            model_answer,
            gold_answer,
            correctness,
            answer_score,
            cot_model_answer,
            cot_correctness,
            cot_score,
            verbalized_cot_qual,
            seq_likelihood,
            cot_seq_likelihood
        ) in zip(
            batch["question_id"],
            batch["question"],
            questions_in_context,
            model_answers,
            batch[answer_column],
            answer_correctness,
            answer_scores,
            cot_model_answers,
            cot_answers_correctness,
            cot_answer_scores,
            verbalized_uncertainties["cot_qual"],
            seq_likelihoods,
            cot_seq_likelihoods,
        ):             
            calibration_data[question_id] = {
                "accuracy": int(correctness),
                "score": float(answer_score),
                "cot_accuracy": int(cot_correctness),
                "cot_score": float(cot_score),
                "gold_answer": gold_answer,
                "answer": model_answer,
                "cot_answer": cot_model_answer,
                "question": question,
                "question_in_context": question_in_context,
                "verbalized_cot_qual": verbalized_cot_qual,
                "seq_likelihood": seq_likelihood.cpu().item(),
                "cot_seq_likelihood": cot_seq_likelihood.cpu().item(),
            }
        
        del outputs

    return calibration_data, included_questions

def create_or_load_calibration_data(
          model: AutoModelForCausalLM,
          tokenizer: AutoTokenizer,
          data_loader: DataLoader,
          device: torch.device | str,
          max_samples: int,
          data_path: str,
          data_dir: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    
    if not os.path.exists(data_path):
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        calibration_data, included_questions = extract_model_calibration_data(
            model=model,
            tokenizer=tokenizer,
            calibration_split=data_loader,
            device=device,
            max_samples=max_samples,
        )
        calibration_data["included_questions"] = included_questions

        with open(data_path, "wb") as calibration_file:
            dill.dump(calibration_data, calibration_file)
        
        del calibration_data["included_questions"]
    else:
        with open(data_path, "rb") as calibration_file:
            calibration_data = dill.load(calibration_file)
        
        included_questions = calibration_data["included_questions"]
        del calibration_data["included_questions"]
    
    return calibration_data, included_questions

########################################
############GENERATE INPUTS#############
########################################

def preprocess_batch_wrapper(
    train_data: Dataset,
    num_in_context_samples: int,
    tokenizer: AutoTokenizer,
    max_input_length: int
) -> Callable:
    """
        Closure for the process batch function that makes a certain variables available to the function scope.

        Loosely based on:
            https://github.com/parameterlab/apricot/blob/main/src/data.py

    """

    def preprocess_batch(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
            Process a specific batch of TriviaQA.
        """
        answer_field = "answer" if "answer" in batch else "best_answer"
        answers = [
            str(answer) if not isinstance(answer, dict) else answer["value"]
            for answer in batch[answer_field]
        ]
        # select few-shot examples
        few_shot_prompts = []
        for _ in range(len(answers)):
            few_shot_prompt = ""

            if num_in_context_samples > 0:
                train_indices = np.random.choice(
                    range(0, len(train_data)), size=num_in_context_samples
                )
                in_context_samples = train_data.select(train_indices)
                for sample in in_context_samples:
                    answer = (
                        sample["answer"]
                        if "answer" in sample and not isinstance(sample["answer"], dict)
                        else sample.get("best_answer", sample["answer"]["value"])
                    )
                    few_shot_prompt += QA_FEW_SHOT_TEMPLATE.format(
                        question=sample["question"], answer=answer
                    )
                
            few_shot_prompts.append(few_shot_prompt)
        
        # Qwen adaptation for English
        batch_with_prompt = [
            few_shot_prompt + " Question: " + question + " Your response should only be in english and no other language." + " Answer: "
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]
        batch_with_cot_prompt = [
            few_shot_prompt + QA_COT_PROMPT + " Question: " + question + " Your response should only be in english and no other language." + " Answer: "
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]
        tokenizer.padding_side = "left" # necessary for left padding
        inputs = tokenizer(
            batch_with_prompt,
            padding="max_length", # for padding, we want to pad the left side of the sequence
            truncation=True, # for truncation, we want to truncate the right side of the sequence
            max_length=max_input_length,
        )
        cot_inputs = tokenizer(
            batch_with_cot_prompt,
            padding="max_length", # for padding, we want to pad the left side of the sequence
            truncation=True, # for truncation, we want to truncate the right side of the sequence
            max_length=max_input_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["cot_input_ids"] = cot_inputs.input_ids
        batch["cot_attention_mask"] = cot_inputs.attention_mask
        batch["answer"] = answers

        # Generate question IDs for OOD test set
        if "question_id" not in batch:
            batch["question_id"] = [
                hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest() 
                for question in batch["question"]
            ] # hash the question to get a unique ID for each question

        return batch
    
    return preprocess_batch

# Create a preprocess_batch_wrapper for TruthfulQA dataset
def preprocess_batch_wrapper_truthful_qa(
    train_data: Dataset,
    num_in_context_samples: int,
    tokenizer: AutoTokenizer,
    max_input_length: int
) -> Callable:
    """
        Closure for the process batch function that makes a certain variables available to the function scope.
    """

    def preprocess_batch(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
            Process a specific batch of TruthfulQA.
        """
        answer_field = "best_answer" if "best_answer" in batch else "answer"
        answers = [
            str(answer) if not isinstance(answer, dict) else answer["value"]
            for answer in batch[answer_field]
        ]
        correct_answers_field = "correct_answers" if "correct_answers" in batch else "answer"
        correct_answers = []
        for batch_answers in batch[correct_answers_field]:
            current_length = len(batch_answers)
            append_batch_answers = batch_answers

            if len(batch_answers) < 12:
                append_batch_answers.extend([""] * (12 - current_length))
            
            correct_answers.append(append_batch_answers)


        # select few-shot examples
        few_shot_prompts = []
        for _ in range(len(answers)):
            few_shot_prompt = ""

            if num_in_context_samples > 0:
                train_indices = np.random.choice(
                    range(0, len(train_data)), size=num_in_context_samples
                )
                in_context_samples = train_data.select(train_indices)

                for sample in in_context_samples:
                    answer = (
                        sample["best_answer"]
                        if "best_answer" in sample and not isinstance(sample["best_answer"], dict)
                        else sample.get("best_answer", sample["best_answer"]["value"])
                    )
                    few_shot_prompt += QA_FEW_SHOT_TEMPLATE.format(question=sample["question"], answer=answer)
                
            few_shot_prompts.append(few_shot_prompt)
        
        batch_with_prompt = [
            few_shot_prompt + " Question: " + question + " Your response should only be in english and no other language." + " Answer:"
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]
        batch_with_cot_prompt = [
            few_shot_prompt + QA_COT_PROMPT + " Question: " + question + " Your response should only be in english and no other language." + " Answer:"
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]
        tokenizer.padding_side = "left" # necessary for left padding
        inputs = tokenizer(
            batch_with_prompt,
            padding="max_length", # for padding, we want to pad the left side of the sequence
            truncation=True, # for truncation, we want to truncate the right side of the sequence
            max_length=max_input_length,
        )
        cot_inputs = tokenizer(
            batch_with_cot_prompt,
            padding="max_length", # for padding, we want to pad the left side of the sequence
            truncation=True, # for truncation, we want to truncate the right side of the sequence
            max_length=max_input_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["cot_input_ids"] = cot_inputs.input_ids
        batch["cot_attention_mask"] = cot_inputs.attention_mask
        batch["best_answer"] = answers
        batch["correct_answers"] = correct_answers

        # Generate question IDs for OOD test set
        if "question_id" not in batch:
            batch["question_id"] = [
                hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest() 
                for question in batch["question"]
            ] # hash the question to get a unique ID for each question

        return batch
    
    return preprocess_batch


def preprocess_trivia_qa(
    model_name: str,
    num_in_context_samples: int,
    batch_size: int,
    data_dir: str,
    validation_fraction: float = 0.01,
    max_input_length: int = MAX_INPUT_LENGTH,
) -> Dict[str, DataLoader]:
    """
    Preprocess the TriviaQA dataset. This involves preparing the inputs by adding a number in-context samples and using
    the target model's tokenizer. This function is loosely based on the code by Lorenz Kuhn:
    https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_triviaqa.py.
    """
    processed_data_dir = os.path.join(
        data_dir, 
        "trivia_qa", 
        model_name.replace("/", "_"),
        "preprocessed_data", 
        f"in_context_{num_in_context_samples}"
    )
    train_data_loader_path = os.path.join(processed_data_dir, "train.dl")
    validation_data_loader_path = os.path.join(processed_data_dir, "validation.dl")
    test_data_loader_path = os.path.join(processed_data_dir, "test.dl")
    # Create directory if it doesn't exist

    if (
        not os.path.exists(train_data_loader_path)
        or not os.path.exists(validation_data_loader_path)
        or not os.path.exists(test_data_loader_path)
    ):
        print("Did not find train data directory, preprocessing trivia_qa.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load data splits - test split does not contain answers, so use (part of) training data for calibration and
        # validation data for testing    
        data_percentage = str(int(100 - validation_fraction * 100))
        train_data = load_dataset(
            "trivia_qa", "rc.nocontext", split=f"train[:{data_percentage}%]"
        )
        validation_data = load_dataset(
            "trivia_qa", "rc.nocontext", split=f"train[{data_percentage}%:]"
        )
        test_data = load_dataset(
            "trivia_qa", "rc.nocontext", split="validation"
        )
        
        if not os.path.exists(processed_data_dir):            
            print(f"Creating the data directory ar {processed_data_dir}")
            os.makedirs(processed_data_dir)
        
        for split_name, data_split, data_loader_path in zip(
            ["train", "validation", "test"], 
            [train_data, validation_data, test_data], 
            [
                train_data_loader_path, 
                validation_data_loader_path, 
                test_data_loader_path,
            ]
        ):
            if os.path.exists(data_loader_path):
                continue

            remove_columns = ["search_results", "question_source", "entity_pages"]

            if split_name == "oodtest":
                remove_columns = ["type", "source"]
            
            data_split = data_split.map(
                # Create processing function by making variables available in the closure
                preprocess_batch_wrapper(
                    train_data=train_data,
                    num_in_context_samples=num_in_context_samples,
                    tokenizer=tokenizer,
                    max_input_length=max_input_length,
                ),
                batched=True, # Add batched=True to process data in batches
                batch_size=batch_size,
                remove_columns=remove_columns,
            )
            data_split.set_format(
                type="torch",
                columns=[
                    "input_ids", 
                    "attention_mask", 
                    "cot_input_ids", 
                    "cot_attention_mask"
                ],
                output_all_columns=True, # This is necessary for the DataLoader to work
            )
            data_loader = DataLoader(
                data_split,
                batch_size=batch_size,
                drop_last=True, # drop last incomplete batch
            ) # This is necessary for the DataLoader to work
            
            data_split.save_to_disk(
                os.path.join(processed_data_dir, split_name + ".data")
            )
            torch.save(
                data_loader,
                os.path.join(processed_data_dir, split_name + ".dl")
            )
    print(f"Loading the data from {train_data_loader_path}")
    data_loaders = {
        "train": torch.load(train_data_loader_path),
        "test": torch.load(test_data_loader_path),
    }

    return data_loaders

# def collated_function(x):
#     """
#     Collate function for DataLoader. This function is responsible for processing the input data and returning a batch.
#     """
#     # if "correct_answers" in x:
#     #     return tuple(x["correct_answers"])
#     input_ids = [item["input_ids"] for item in x]
#     attention_masks = [item["attention_mask"] for item in x]
#     cot_input_ids = [item["cot_input_ids"] for item in x]
#     cot_attention_masks = [item["cot_attention_mask"] for item in x]

#     return {
#         "input_ids": torch.tensor(input_ids),
#         "attention_mask": torch.tensor(attention_masks),
#         "cot_input_ids": torch.tensor(cot_input_ids),
#         "cot_attention_mask": torch.tensor(cot_attention_masks),
#         "type": [item["type"] for item in x],
#         "category":[item["category"] for item in x],
#         "question": [item['question'] for item in x],
#         "best_answer": [item['best_answer'] for item in x],
#         "correct_answers": [item['correct_answers'] for item in x],
#         "question_id": [item['question_id'] for item in x],
#     }

def preprocess_truthful_qa(
    model_name: str,
    num_in_context_samples: int,
    batch_size: int,
    data_dir: str,
    validation_fraction: float = 0.01,
    max_input_length: int = MAX_INPUT_LENGTH,
) -> Dict[str, DataLoader]:
    """
    Preprocess the TruthfulQA dataset. This involves preparing the inputs by adding a number in-context samples and using
    the target model's tokenizer. This function is loosely based on the code by Lorenz Kuhn:
    https://github.com/lorenzkuhn/semantic_uncertainty/blob/main/code/parse_triviaqa.py.
    """
    processed_data_dir = os.path.join(
        data_dir, 
        "truthful_qa", 
        model_name.replace("/", "_"),
        "preprocessed_data", 
        f"in_context_{num_in_context_samples}"
    )
    train_data_loader_path = os.path.join(processed_data_dir, "train.dl")
    test_data_loader_path = os.path.join(processed_data_dir, "test.dl")

    if (
        not os.path.exists(train_data_loader_path) 
        or not os.path.exists(test_data_loader_path)
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: # Mistral 
            tokenizer.pad_token = tokenizer.eos_token

        # Load data splits - test split does not contain answers, so use (part of) training data for calibration and
        # validation data for testing
        train_data = load_dataset("truthful_qa", "generation", split="validation[50%:]")
        test_data = load_dataset("truthful_qa", "generation", split="validation[:50%]")

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        
        for split_name, data_split, data_loader_path in zip(
            ["train", "test"], 
            [train_data, test_data], 
            [train_data_loader_path, test_data_loader_path],
        ):
        
            if os.path.exists(data_loader_path):
                continue

            remove_columns = ["incorrect_answers", "source"]

            data_split = data_split.map(
                # Create processing function by making variables available in the closure
                preprocess_batch_wrapper_truthful_qa(
                    train_data=train_data,
                    num_in_context_samples=num_in_context_samples,
                    tokenizer=tokenizer,
                    max_input_length=max_input_length,
                ),
                batched=True, # Add batched=True to process data in batches
                batch_size=batch_size,
                remove_columns=remove_columns,
            )
            data_split.set_format(
                type="torch",
                columns=[
                    "input_ids", 
                    "attention_mask", 
                    "cot_input_ids", 
                    "cot_attention_mask"
                ],
                output_all_columns=True, # This is necessary for the DataLoader to work
            )
            data_loader = DataLoader(
                data_split,
                batch_size=batch_size,
                drop_last=True, # drop last incomplete batch
             ) # This is necessary for the DataLoader to work
            data_split.save_to_disk(
                os.path.join(processed_data_dir, split_name + ".data")
            )
            torch.save(
                data_loader,
                os.path.join(processed_data_dir, split_name + ".dl")
            )
        
    data_loaders = {
        "train": torch.load(train_data_loader_path),
        "test": torch.load(test_data_loader_path),
    }

    return data_loaders

def preprocess_natural_questions(
    model_name: str,
    num_in_context_samples: int,
    batch_size: int,
    data_dir: str,
    validation_fraction: float = 0.01,
    max_input_length: int = MAX_INPUT_LENGTH,
) -> Dict[str, DataLoader]:
    """
    Preprocess the Natural Questions dataset. This involves preparing the inputs by adding a number in-context samples and using
    the target model's tokenizer. The function handles both long answers (paragraphs) and short answers (entities/spans).
    """
    processed_data_dir = os.path.join(
        data_dir, 
        "natural_questions", 
        model_name.replace("/", "_"),
        "preprocessed_data", 
        f"in_context_{num_in_context_samples}"
    )
    train_data_loader_path = os.path.join(processed_data_dir, "train.dl")
    test_data_loader_path = os.path.join(processed_data_dir, "test.dl")

    if (
        not os.path.exists(train_data_loader_path)
        or not os.path.exists(test_data_loader_path)
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load Natural Questions dataset
        train_data = load_dataset("nq_open", split="train")
        test_data = load_dataset("nq_open", split="validation")

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        
        for split_name, data_split, data_loader_path in zip(
            ["train", "test"], 
            [train_data, test_data], 
            [train_data_loader_path, test_data_loader_path],
        ):
            if os.path.exists(data_loader_path):
                continue

            # NQ-open only has question and answer columns, no need to remove anything
            remove_columns = []

            data_split = data_split.map(
                # Create processing function by making variables available in the closure
                preprocess_batch_wrapper_natural_questions(
                    train_data=train_data,
                    num_in_context_samples=num_in_context_samples,
                    tokenizer=tokenizer,
                    max_input_length=max_input_length,
                ),
                batched=True,
                batch_size=batch_size,
                remove_columns=remove_columns,
            )
            data_split.set_format(
                type="torch",
                columns=[
                    "input_ids", 
                    "attention_mask", 
                    "cot_input_ids", 
                    "cot_attention_mask"
                ],
                output_all_columns=True,
            )
            data_loader = DataLoader(
                data_split,
                batch_size=batch_size,
                drop_last=True,
            )
            
            data_split.save_to_disk(
                os.path.join(processed_data_dir, split_name + ".data")
            )
            torch.save(
                data_loader,
                os.path.join(processed_data_dir, split_name + ".dl")
            )
    
    data_loaders = {
        "train": torch.load(train_data_loader_path),
        "test": torch.load(test_data_loader_path),
    }

    return data_loaders

def preprocess_batch_wrapper_natural_questions(
    train_data: Dataset,
    num_in_context_samples: int,
    tokenizer: AutoTokenizer,
    max_input_length: int
) -> Callable:
    """
    Closure for the process batch function that makes certain variables available to the function scope.
    Specifically handles NQ-open format with direct question-answer pairs.
    """

    def preprocess_batch(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a specific batch of NQ-open questions.
        """
        # Generate question IDs if not present
        if "question_id" not in batch:
            batch["question_id"] = [
                hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest() 
                for question in batch["question"]
            ]

        # Select few-shot examples
        few_shot_prompts = []
        for _ in range(len(batch["question"])):
            few_shot_prompt = ""

            if num_in_context_samples > 0:
                train_indices = np.random.choice(
                    range(0, len(train_data)), size=num_in_context_samples
                )
                in_context_samples = train_data.select(train_indices)
                
                for sample in in_context_samples:
                    # For NQ-open, we use the first answer if multiple are available
                    sample_answer = sample["answer"][0] if sample["answer"] else "NULL"
                    few_shot_prompt += QA_FEW_SHOT_TEMPLATE.format(
                        question=sample["question"], 
                        answer=sample_answer
                    )
            
            few_shot_prompts.append(few_shot_prompt)
        
        # Create prompts with context
        batch_with_prompt = [
            few_shot_prompt + " Question: " + question + " Your response should only be in english and no other language." + " Answer: "
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]
        
        # Create prompts with Chain-of-Thought
        batch_with_cot_prompt = [
            few_shot_prompt + QA_COT_PROMPT + " Question: " + question + " Your response should only be in english and no other language." + " Answer: "
            for question, few_shot_prompt in zip(batch["question"], few_shot_prompts)
        ]

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_with_prompt,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )
        cot_inputs = tokenizer(
            batch_with_cot_prompt,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )

        # For NQ-open, use the first answer if multiple are available
        batch["answer"] = [ans[0] if ans else "NULL" for ans in batch["answer"]] 
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["cot_input_ids"] = cot_inputs.input_ids
        batch["cot_attention_mask"] = cot_inputs.attention_mask

        return batch
    
    return preprocess_batch

def load_experiment_dataset(
    model_name: str,
    dataset_name: str,
    num_in_context_samples: int,
    batch_size: int,
    data_dir: str,
) -> Dict[str, DataLoader]:
    # Load data
    """
    Parameters
    ----------
    model_name: str
        Identifier (huggingface or openai) for the large model
    dataset_name: str
        Name of the target dataset
    num_in_context_samples: int
        Number of in-context learning samples to include in the prompt.
    batch_size: int
        Batch  size
    data_dir: str
        Directory the contains dataset

    Returns
    -------
    Dict[str, Dataloader]
        Dataloaders of the validation and test sets
    """
    assert (
        dataset_name in DATASETS
    ), f"dataset should be one of the {' ,'.join(DATASETS)}"

    if dataset_name == "trivia_qa":
        return preprocess_trivia_qa(
            model_name=model_name,
            num_in_context_samples=num_in_context_samples,
            batch_size=batch_size,
            data_dir=data_dir,
        )
    elif dataset_name == "truthful_qa":
        return preprocess_truthful_qa(
            model_name=model_name,
            num_in_context_samples=num_in_context_samples,
            batch_size=batch_size,
            data_dir=data_dir,
        )
    elif dataset_name == "natural_questions":
        return preprocess_natural_questions(
            model_name=model_name,
            num_in_context_samples=num_in_context_samples,
            batch_size=batch_size,
            data_dir=data_dir,
        )
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} not supported. Please add custom code for preprocessing."
        )
