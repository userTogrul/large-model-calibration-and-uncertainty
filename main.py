"""
    Main integration script.
"""
# STANDARD LIB
import argparse # add arguments in the command prompt
from collections import defaultdict
from copy import deepcopy
import os # get the API key with that
import warnings # showing warnings
import dill
import glob # to check the batch files
import gc
# EXTERNAL LIB
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from openai import OpenAI, BadRequestError
from transformers import (
     AutoModelForCausalLM,
     AutoTokenizer,
     AutoConfig,
     AutoModelForQuestionAnswering,
     set_seed,
)
import pandas as pd
import numpy as np
from datasets import load_dataset
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import wandb
from wandb.sdk.wandb_run import Run as wandb_Run
# constants
from src.constant_vals import (
     EMBEDDING_MODEL,
     SENTENCE_EMBEDDING_MODEL,
     GPT_MODEL,
     LLM_MODEL,
     EMISSION_DIR,
     PROJECT_NAME,
     RESULT_DIR,
     HF_HOME,
     DATA_DIR,
     IMG_DIR,
     SEED,
     ALLOWED_INPUTS,
     DATASETS,
     BASELINE_METHODS,
     CALIBRATION_BATCH_SIZE,
     CALIBRATION_LEARNING_RATE,
     CALIBRATION_NUM_STEPS,
     CALIBRATION_VALID_INTERVAL,
     DATASET_SPLIT_SIZES,
     NUM_IN_CONTEXT_SAMPLES,
     INPUT_PARTS,
     CLOSED_BOX_MODELS,
     QUALITATIVE_SCALE,
     EVAL_METRIC_ORDER,
     MAX_INPUT_LENGTH,
     SEMANTIC_ENTROPY_NUM_SAMPLES,
     SEMANTIC_ENTROPY_TEMPERATURE,
     SEMANTIC_ENTROPY_NLI_MODEL,
)
from methods.platt_scaling.platt_scaling import PlattScaling
from methods.temperature_scaling.temp_scaling import TemperatureScaling
from sklearn.isotonic import IsotonicRegression # isotonic regression
from src.data import load_experiment_dataset, unpack_dataloader, create_or_load_calibration_data, extract_closed_box_calibration_data
from src.evaluation import (
    get_target_function, 
    extract_verbalized_confidence,
    evaluate_confidences,
)
from src.plots import plot_reliability_diagram, plot_reliability_diagram_cdf
from methods.hallumeasure.hallumeasure import HalluMeasure
from methods.lmvslm import LMvsLM
from methods.ourmethod import ClaimUncertaintyMeasurer
from methods.semantic_entropy import SemanticEntropy
import csv
from tqdm import tqdm
# from pudb import set_trace; set_trace() # for debugging

# SECRETS
SECRET_IMPORTED = False
try:
     from secret import OPENAI_API_KEY, TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE, WANDB_API_KEY, HF_TOKEN, BASE_URL
     SECRET_IMPORTED = True
     os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
     os.environ["WANDB_API_KEY"] = WANDB_API_KEY
     os.environ['HF_TOKEN'] = HF_TOKEN
     os.environ["BASE_URL"] = BASE_URL

except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("secret.py could not be imported.")
    try:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        TELEGRAM_API_TOKEN = os.environ.get("TELEGRAM_API_TOKEN")
        TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
        COUNTRY_CODE = os.environ.get("COUNTRY_CODE")
        WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
        BASE_URL = os.environ.get("BASE_URL")
        SECRET_IMPORTED = True
    except AttributeError:
        raise ImportError(
                "secret.py wasn't found, please create this file and add necessary information or"
                "add environment variables using os.environ."
        )

# HF
os.environ['HF_HOME'] = HF_HOME
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
     torch.cuda.manual_seed(SEED)
set_seed(SEED)

def tune_alpha(
    ver_scores: np.ndarray,
    freq_scores: np.ndarray,
    correctness: np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Tuple[float, Dict[float, float]]:
    """Grid-search alpha on a held-out split, minimising ECE.

    Args:
        ver_scores: Per-example verification-only confidence scores.
        freq_scores: Per-example consistency (freq) confidence scores.
        correctness: Binary correctness labels (0/1).
        alpha_grid: Candidate alpha values (default: 0.0, 0.1, …, 1.0).
        n_bins: Number of ECE bins.

    Returns:
        (best_alpha, ece_by_alpha) where ece_by_alpha maps each alpha to its ECE.
    """
    from methods.ece.ECE import ece as compute_ece
    from src.evaluation import get_target_function

    if alpha_grid is None:
        alpha_grid = np.round(np.arange(0.0, 1.01, 0.1), 1)

    best_alpha = 0.0
    best_ece = float("inf")
    ece_by_alpha: Dict[float, float] = {}

    for alpha in alpha_grid:
        combined = alpha * freq_scores + (1.0 - alpha) * ver_scores
        combined = np.clip(combined, 0.0, 1.0)
        try:
            target_func = get_target_function(combined, correctness, n_bins)
            targets = target_func(combined)
            val = float(compute_ece(y_true=targets, y_pred=combined, n_bins=n_bins))
        except Exception:
            val = float("inf")
        ece_by_alpha[round(float(alpha), 1)] = round(val, 4)
        if val < best_ece:
            best_ece = val
            best_alpha = round(float(alpha), 1)

    return best_alpha, ece_by_alpha


def run_calibration_benchmark(
    model_name: str,
    baseline_methods: List[str],
    dataset_name: str,
    num_in_context_samples: int,
    use_binary_targets: bool,
    input_parts: List[str],
    device: str,
    data_dir: str,
    seed: int,
    calibration_batch_size: Optional[int] = None,
    calibration_learning_rate: Optional[int] = None,
    calibration_num_steps: Optional[int] = None,
    calibration_valid_interval: Optional[int] = None,
    img_dir: Optional[str] = None,
    result_dir: Optional[str]=None,
    wandb_run: Optional[wandb_Run] = None
):
    # TODO
    """
        model_name: str
            LLM name.
        baseline_methods: List[str]
            Calibration method name.
        dataset_name: str
            Name of the dataset to be used.
        num_in_context_samples: int
            Number of in-context samples to add to the prompt.
        calibration_batch_size: Optional[int]
            Batch size used for Platt Scaling and Temperature scaling. Default is none.
        calibration_learning_rate: Optional[int]
            Learning rate used for Platt Scaling and Temperature scaling. Default is none.
        calibration_num_steps: Optional[int]
            Number of steps used for Platt Scaling and Temperature scaling. Default is none.
        calibration_valid_interval: Optional[int]
            Size of interval used for validation used for Platt Scaling and Temperature scaling. Default is none.
        use_binary_targets: bool
            Weather to use binary targets for measuring calibration.
        input_parts: List[str]
            Which parts of the input to use. This includes 'question', 'answer', 'cot_answer', 'quantitative',
            'cot_quantitative', 'qualitative' and 'cot_qualitative'. Used for ablation study.
        device: str
            GPU support.
        data_dir: str
            Path to the directory contatining data.
        img_dir: Optional[str]
            Path to save the plotted reliability diagrams. Default is None.
        result_dir: Optional[str]
            Path to save the baseline results. Default is None.
        wandb_run: wandb_Run
            Weights & Biases run to log results. Default is None.
    """
    """
        Source:
        https://github.com/parameterlab/apricot/blob/main/compute_baselines.py
    """
    # Function for executing calibration experiments
    assert(
         "answer" not in input_parts or "cot_answer" not in input_parts
    ), "Choose either 'answer' or 'cot_answer' for input_parts', not both."
    assert(
         "qualitative" not in input_parts or "quantitative" not in input_parts
    ), "Choose either 'qualitative' or 'cot_anwer' for 'quantitative', not both."
    ##########

    input_parts = list(sorted(input_parts))
    suffix = "_".join(input_parts)
    # Calibration dataloader path
    calibration_data_dir = os.path.join(
         data_dir,
         dataset_name,
         model_name.replace("/", "_"),
         "calibration_data",
         f"in_context_{num_in_context_samples}",
     ) # ?
    # calibration_target_file = os.path.join(calibration_data_dir, "calibration_targets.dill")
    timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
    calibration_data, included_questions = {}, {} # dictionary to hold the data for calibration
    # Pre-process the data
    data_split_names = list(DATASET_SPLIT_SIZES[dataset_name].keys())

    if (
         any(
              [ 
                   not os.path.exists(
                        os.path.join(
                             calibration_data_dir, 
                             f"calibration_data_{split_name}.dill"
                         )
                    ) 
                    for split_name in data_split_names 
               ]  
            )
     ):
         
         if model_name not in CLOSED_BOX_MODELS:
              # get the calibration data
              data_loaders = load_experiment_dataset(
                   model_name = model_name,
                   dataset_name = dataset_name,
                   num_in_context_samples = num_in_context_samples,
                   batch_size=calibration_batch_size,
                   data_dir = data_dir
              )
              # LOAD MODEl
              config = AutoConfig.from_pretrained(model_name)
              config.max_length = 550
              model = AutoModelForCausalLM.from_pretrained(
                   model_name, 
                   torch_dtype=torch.float16, 
                   low_cpu_mem_usage=True, 
                   config=config,
              ).to(device)
              tokenizer = AutoTokenizer.from_pretrained(model_name, padding='max_length', padding_side="left", max_length=MAX_INPUT_LENGTH)

              if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    model.config.pad_token_id = tokenizer.eos_token_id

              # Unpack the data loaders
              for split in data_split_names:
                   calibration_data_path = os.path.join(calibration_data_dir, f"calibration_data_{split}.dill")
                   (
                        calibration_data[split],
                        included_questions[split],
                   ) = create_or_load_calibration_data(
                        model=model,
                        tokenizer=tokenizer,
                        data_loader=data_loaders[split],
                        device=device,
                        max_samples=DATASET_SPLIT_SIZES[dataset_name][split],
                        data_path=calibration_data_path,
                        data_dir=calibration_data_dir,
                   )
               # outside for loop
               # Free up memory after loading calibration data
              del model, data_loaders, tokenizer
              if torch.cuda.is_available():
                    torch.cuda.empty_cache()
              gc.collect()
              pass
         else:
              for split in data_split_names:
                    calibration_data[split] = extract_closed_box_calibration_data(
                         model_name=model_name,
                         split=split, 
                         dataset_name=dataset_name,
                         num_in_context_samples=num_in_context_samples,
                         data_dir=data_dir, 
                    )
              pass
    else:
         # DOUBLE CHECK
         for split in data_split_names:
              split_data = dill.load(
                   open(os.path.join(calibration_data_dir, f"calibration_data_{split}.dill"), "rb")
              )

              if "included_questions" in split_data:
                   del split_data["included_questions"] # TODO: Remove this line after testing

              calibration_data[split] = split_data
         pass
    
    # Train parametric scaling and nonparametric methods for calibration
    parametric_scalers = {}
    nonparametric_scalers = {}
    for method in ["ps_seq_likelihood", "ts_seq_likelihood", "ir_seq_likelihood"]:
         
         if method in baseline_methods:
               train_likelihood_name = "seq_likelihood"
               train_likelihoods = np.array(
                    [
                         question_data[train_likelihood_name]
                         for question_data in calibration_data["train"].values()
                    ]
               )
               train_likelihoods[np.isnan(train_likelihoods)] = 0
               train_correctness = [
                    question_data["accuracy"] for question_data in calibration_data["train"].values()
               ]
               # Compute targets or accuracy per bin
               train_target_func = get_target_function(
                    train_likelihoods, 
                    train_correctness
               )

               train_likelihoods = torch.FloatTensor(train_likelihoods)
               train_targets = torch.FloatTensor(train_target_func(train_likelihoods))
               
               # Platt Scaling
               if method == "ps_seq_likelihood":
                    print(f"Doing platt scaling for {method}")
                    scaler = PlattScaling()
                    scaler.train_parameters(
                         train_probabilities=train_likelihoods,
                         train_targets=train_targets,
                         batch_size=calibration_batch_size,
                         learning_rate=calibration_learning_rate,
                         num_steps=calibration_num_steps,
                    )
                    parametric_scalers[method] = scaler   

               # Temp Scaling
               if method == "ts_seq_likelihood":
                    print(f"Doing temperature scaling for {method}")
                    scaler = TemperatureScaling()
                    scaler.train_parameters(
                         train_probabilities=train_likelihoods,
                         train_targets=train_targets,
                         batch_size=calibration_batch_size, # use fullbatch for L-BFGS 
                         learning_rate=calibration_learning_rate,
                         num_steps=calibration_num_steps,
                    )
                    parametric_scalers[method] = scaler

               # Isotonic regression
               if method == "ir_seq_likelihood":
                    print(f"Doing isotonic regression for {method}")
                    scaler = IsotonicRegression(out_of_bounds='clip')
                    scaler.fit(
                         train_likelihoods,
                         train_targets,
                    )
                    nonparametric_scalers[method] = scaler
                    
    ########################################
    ## Alpha tuning for ourmethod (train) ##
    ########################################
    # Tune alpha = weight of freq(c_i) vs ver(c_i) on the held-out train split.
    # Held-out sizes: TruthfulQA train = 409; TriviaQA train 20% ≈ 2,400 examples.
    ourmethod_best_alpha: float = 0.0   # default: verification-only (matches current results)
    ourmethod_alpha_sensitivity: Dict[float, float] = {}

    if "ourmethod" in baseline_methods:
        print(f"\n[ourmethod] Tuning alpha on train split ({dataset_name}, {model_name}) …")
        openai_client_tune = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        measurer_tune = ClaimUncertaintyMeasurer(
            model_name=model_name,
            device=device,
            claim_extractor_model="gpt-4o-mini",
        )
        train_split_data = calibration_data["train"]
        # For TriviaQA use 20 % of train; for TruthfulQA use the full 409-example train split
        tune_fraction = 0.1 if dataset_name == "trivia_qa" else 1.0
        tune_items = list(train_split_data.items())
        n_tune = max(1, int(len(tune_items) * tune_fraction))
        rng = np.random.default_rng(seed)
        tune_idx = rng.choice(len(tune_items), size=n_tune, replace=False)
        tune_items = [tune_items[i] for i in tune_idx]

        train_ver, train_freq, train_correct = [], [], []
        for _, question_data in tqdm(tune_items, desc="[alpha-tune] computing ver+freq"):
            answer = question_data.get("answer", "")
            question = question_data.get("question", "")
            accuracy = float(question_data.get("accuracy", 0))
            try:
                claims = measurer_tune.measure_uncertainty_for_answer(
                    answer=answer,
                    question=question,
                    openai_client=openai_client_tune,
                    alpha=0.0,               # store raw ver and freq separately
                    n_consistency_samples=5,
                )
                if claims:
                    train_ver.append(float(np.mean([c.ver_confidence for c in claims])))
                    train_freq.append(float(np.mean([c.freq_confidence for c in claims])))
                else:
                    train_ver.append(0.5)
                    train_freq.append(0.5)
            except Exception as e:
                print(f"[alpha-tune] error: {e}")
                train_ver.append(0.5)
                train_freq.append(0.5)
            train_correct.append(accuracy)

        ourmethod_best_alpha, ourmethod_alpha_sensitivity = tune_alpha(
            ver_scores=np.array(train_ver),
            freq_scores=np.array(train_freq),
            correctness=np.array(train_correct),
        )
        print(
            f"[ourmethod] best α = {ourmethod_best_alpha}  "
            f"(model={model_name}, dataset={dataset_name})"
        )
        print(f"[ourmethod] ECE by α: {ourmethod_alpha_sensitivity}")

    ################################
    ## Evaluate baseline methods  ##
    ################################
    baseline_confidences = defaultdict(dict) #
    baseline_results = {}
    masks = defaultdict(dict) # For qualitative verbalization

    for method in baseline_methods:
         for split_name in data_split_names:
              
              if "test" not in split_name: # Only evaluate on the test split
                   continue
              
              split_data = calibration_data[split_name]

              # Extract confidence scores
              likelihoods = np.array(
                   [
                        question_data["seq_likelihood"]
                        for question_data in split_data.values()
                   ]
              )
              likelihoods[np.isnan(likelihoods)] = 0
              baseline_confidences[split_name]["seq_likelihood"] = likelihoods
              # Initialize HalluMeasure and LMvsLM if needed
              hallumeasure = None
              lmvslm = None

              # Add HalluMeasure evaluation
              if method == "hallumeasure":
                  hallumeasure = HalluMeasure(device=device)
                  # Initialize OpenAI client
                  openai_client = OpenAI(
                         api_key=OPENAI_API_KEY,
                         base_url=BASE_URL,
                  )
                  hallucination_scores = []
                  for question_data in tqdm(split_data.values(), desc=f"Processing {method} for {split_name}"):
                      try:
                         response = question_data["answer"]
                         context = question_data["gold_answer"]
                         result = hallumeasure.measure_hallucination(response, context, openai_client)
                         hallucination_scores.append(1.0 - result.hallucination_score)  # Convert to confidence score
                      except BadRequestError as e:
                         print(f"Response: {response}")
                         print(f"Context: {context}")
                         hallucination_scores.append(0.0)
                         continue
                      except Exception as e:
                         print(f"Error: {e}")
                         hallucination_scores.append(0.0)
                         continue
                  baseline_confidences[split_name]["hallumeasure"] = np.array(hallucination_scores)
               
              if method == "semantic_entropy":
                  # Kuhn et al. 2023 / Farquhar et al. 2024:
                  # K-sample generation + bidirectional NLI clustering + cluster-marginal entropy.
                  se = SemanticEntropy(
                       nli_model_name=SEMANTIC_ENTROPY_NLI_MODEL,
                       device=device,
                       num_samples=SEMANTIC_ENTROPY_NUM_SAMPLES,
                       generation_temperature=SEMANTIC_ENTROPY_TEMPERATURE,
                       seed=seed,
                  )
                  se_scores = []
                  if model_name in CLOSED_BOX_MODELS:
                       openai_client_se = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
                       for question_data in tqdm(split_data.values(), desc=f"Processing {method} for {split_name}"):
                           try:
                               result = se.measure(
                                   prompt=question_data["question"],
                                   question=question_data["question"],
                                   openai_client=openai_client_se,
                                   model_name=model_name,
                               )
                               se_scores.append(float(result.confidence))
                           except Exception as e:
                               print(f"Error (semantic_entropy, closed-box): {e}")
                               se_scores.append(0.0)
                  else:
                       config = AutoConfig.from_pretrained(model_name)
                       config.max_length = 550
                       se_model = AutoModelForCausalLM.from_pretrained(
                           model_name,
                           torch_dtype=torch.float16,
                           low_cpu_mem_usage=True,
                           config=config,
                       ).to(device)
                       se_tokenizer = AutoTokenizer.from_pretrained(
                           model_name, padding="max_length", padding_side="left",
                           max_length=MAX_INPUT_LENGTH,
                       )
                       if se_tokenizer.pad_token is None:
                           se_tokenizer.pad_token = se_tokenizer.eos_token
                           se_model.config.pad_token_id = se_tokenizer.eos_token_id

                       for question_data in tqdm(split_data.values(), desc=f"Processing {method} for {split_name}"):
                           try:
                               result = se.measure(
                                   prompt=question_data["question"],
                                   question=question_data["question"],
                                   model=se_model,
                                   tokenizer=se_tokenizer,
                               )
                               se_scores.append(float(result.confidence))
                           except Exception as e:
                               print(f"Error (semantic_entropy, open-box): {e}")
                               se_scores.append(0.0)
                       del se_model, se_tokenizer
                       if torch.cuda.is_available():
                           torch.cuda.empty_cache()
                       gc.collect()
                  baseline_confidences[split_name]["semantic_entropy"] = np.array(se_scores)

              if method == "lmvslm":
                  lmvslm = LMvsLM(examinee_model=model_name, device=device)
                  lmvslm_scores = []
                  for question_data in tqdm(split_data.values(), desc=f"Processing {method} for {split_name}"):
                      try:
                         claim = f"Question: {question_data['question']}\nAnswer: {question_data['answer']}"
                         lmvslm_confidence = lmvslm.get_confidence(claim)
                         lmvslm_scores.append(lmvslm_confidence)
                      except BadRequestError as e:
                         print(f"Claim: {claim}")
                         lmvslm_scores.append(0.0)
                      except Exception as e:
                         print(f"Error: {e}")
                         lmvslm_scores.append(0.0)
                  baseline_confidences[split_name]["lmvslm"] = np.array(lmvslm_scores)
                  # Free up memory after loading calibration data
                  lmvslm.__del__()
               # TODO: run this lines
              if method == "ourmethod":
                  openai_client = OpenAI(
                         api_key=OPENAI_API_KEY,
                         base_url=BASE_URL,
                  )
                  measurer = ClaimUncertaintyMeasurer(
                         model_name=model_name,
                         device=device,
                         claim_extractor_model="gpt-4o-mini"
                   )
                  ourmethod_scores = []
                  print(
                      f"Processing {method} for {split_name} "
                      f"(α={ourmethod_best_alpha}, N=5 consistency samples)"
                  )
                  for question_data in tqdm(split_data.values()):
                         answer = question_data["answer"]
                         question = question_data["question"]
                         claims_with_uncertainty = measurer.measure_uncertainty_for_answer(
                              answer=answer,
                              question=question,
                              openai_client=openai_client,
                              alpha=ourmethod_best_alpha,
                              n_consistency_samples=5,
                         )
                         for i, claim in enumerate(claims_with_uncertainty, 1):
                              print(f"\nClaim {i}:")
                              print(f"  Text: {claim.text}")
                              print(
                                  f"  ver={claim.ver_confidence:.4f}  "
                                  f"freq={claim.freq_confidence:.4f}  "
                                  f"combined={claim.uncertainty:.4f}"
                              )
                              print(f"  Number of tokens: {claim.num_tokens}")

                         # Get overall answer confidence (mean of combined per-claim scores)
                         overall_uncertainty = measurer.get_aggregated_uncertainty(
                              aggregation="mean",
                              claims_with_uncertainty=claims_with_uncertainty,
                         )
                         print(f"Overall answer confidence: {overall_uncertainty:.4f}")
                         ourmethod_scores.append(overall_uncertainty)
                  baseline_confidences[split_name]["ourmethod"] = np.array(ourmethod_scores)
                  # Store alpha metadata in results
                  baseline_results["ourmethod_best_alpha"] = ourmethod_best_alpha
                  baseline_results["ourmethod_alpha_sensitivity"] = str(ourmethod_alpha_sensitivity)
                  
              cot_likelihoods = np.array(
                   [
                        question_data["cot_seq_likelihood"]
                        for question_data in split_data.values()
                   ]
              )
              cot_likelihoods[np.isnan(cot_likelihoods)] = 0
              baseline_confidences[split_name]["cot_seq_likelihood"] = cot_likelihoods

              # Parametric scaling methods
              if method in ["ps_seq_likelihood", "ts_seq_likelihood", "ir_seq_likelihood"]:
                   
                   if method == "ir_seq_likelihood":
                        scaler = nonparametric_scalers[method]
                   else:
                        scaler = parametric_scalers[method]
                   
                   inputs = torch.FloatTensor(likelihoods)
                   outputs = []
                   # TODO
                   with torch.no_grad():
                        
                        if method == "ps_seq_likelihood":
                             outputs = scaler.forward(inputs).numpy()
                        elif method == "ts_seq_likelihood":
                             outputs = np.array(scaler.forward(inputs))
                        elif method == "ir_seq_likelihood":
                             outputs = np.array(scaler.transform(inputs))

                   baseline_confidences[split_name][method] = outputs
                   # Compute accuracy
                   accuracy = np.mean(
                        [
                             question_data["accuracy"]
                             for question_data in split_data.values()
                        ]
                   )
                   baseline_results[f"{method}"] = accuracy
                   
               # Convert verbalized uncertainties into probabilities
              infix = "_cot" if "cot" in method else ""

              if "qual" in method:
                    qual_uncertainties = [
                         question_data[f"verbalized{infix}_qual"]
                         for question_data in split_data.values()
                    ]
                    confidences, successes = extract_verbalized_confidence(
                         qual_uncertainties,
                         mode="qualitative",
                         expression_mapping=QUALITATIVE_SCALE,
                    )
                    baseline_confidences[split_name][
                         f"verbalized{infix}_qual"
                    ] = confidences
                    masks[split_name][f"verbalized{infix}_qual"] = np.array(successes)
               

    #TODO: SFT later

    #### EVALUATION ###
    eval_data = defaultdict(lambda: dict())
    for split_name in data_split_names:
         cdf_baseline_confidences = {}
         cdf_baseline_correctness = {}
         for baseline_name, confidences in baseline_confidences[split_name].items():
              correctness = np.array(
                   [
                        question_data["accuracy"]
                        for question_data in calibration_data[split_name].values()
                   ]
               )
              eval_score = np.array(
                    [
                         question_data["score"]
                         for question_data in calibration_data[split_name].values()
                    ]
               )
              
              if "cot" in baseline_name:
                    correctness = np.array(
                         [
                              question_data["cot_accuracy"]
                              for question_data in calibration_data[split_name].values()
                         ]
                    )
                    eval_score = np.array(
                         [
                              question_data["cot_score"]
                              for question_data in calibration_data[split_name].values()
                         ]
                    )

              if baseline_name in masks[split_name]:
                   baseline_mask = masks[split_name][baseline_name]
                   baseline_results[f"{split_name}_{baseline_name}_success"] = np.mean(baseline_mask.astype(int))
                   correctness = correctness[baseline_mask]
          
              baseline_res = evaluate_confidences(
                    split_name=split_name,
                    add_name=baseline_name,
                    all_confidences=confidences,
                    all_correctness=correctness,
                    all_scores=eval_score,
               )
              baseline_results.update(baseline_res)
              eval_data[baseline_name][split_name] = {
                   "all_confidences": confidences,
                   "all_correctness": correctness,
              }

              # Collect data for combined reliability diagrams
              if img_dir is not None:
                    if not os.path.exists(img_dir):                                                                                                                                      
                        os.makedirs(img_dir)
                    
                    model_name_edited = model_name.replace("/", "_")
                   
                    plot_reliability_diagram(
                         confidences,
                         correctness,
                         save_path=os.path.join(
                              img_dir,
                              f"{split_name}_{baseline_name}_{dataset_name}_{model_name_edited}.pdf",
                         ),
                         success_percentage=baseline_results.get(
                              f"{split_name}_{baseline_name}_success", 1
                         ),
                    )
                    cdf_baseline_confidences[baseline_name] = confidences
                    cdf_baseline_correctness[baseline_name] = correctness

         if img_dir is not None and len(cdf_baseline_confidences) > 0:
              if not os.path.exists(img_dir):
                   os.makedirs(img_dir)
              model_name_edited = model_name.replace("/", "_")
              plot_reliability_diagram_cdf(
                   baseline_confidences=cdf_baseline_confidences,
                   save_path=os.path.join(
                        img_dir,
                        f"{split_name}_all_baselines_cdf_{dataset_name}_{model_name_edited}.pdf",
                   ),
                   baseline_correctness=cdf_baseline_correctness,
              )
              
    # Create results directory to save results
    baseline_results_dir=None
    
    if result_dir is not None:
         baseline_results_dir = os.path.join(
              result_dir,
              dataset_name,
              model_name.replace("/", "_"), # ?
              f"in_context_{num_in_context_samples}",
         )

    os.makedirs(baseline_results_dir, exist_ok=True)

    for baseline_name, baseline_data in eval_data.items():
         with open(
              os.path.join(
                   baseline_results_dir, 
                   f"{timestamp}_{baseline_name}_results.dill",
               ), 
              "wb",
         ) as results_file:
              dill.dump(
                   {
                        "info": {
                             "timestamp": timestamp,
                             "dataset_name": dataset_name,
                             "model_name": model_name,
                             "baseline_name": baseline_name,
                        },
                        "eval_data": baseline_data,
                   },
                   results_file,
              )
    results_df = pd.DataFrame(columns=EVAL_METRIC_ORDER)
    # CREATE RESULTS DATAFRAME
    test_splits = [
         split for split in calibration_data.keys() if "test" in split
    ]
    for baseline_name in baseline_methods:
         for name, result in baseline_results.items():
              for eval_metric in EVAL_METRIC_ORDER:
                   
                   if f"_{eval_metric}" in name and any(
                        [f"{split}_{baseline_name}" in name for split in test_splits]
                   ):
                        results_df.at[baseline_name, eval_metric] = round(result, 3)
                        break

    print(results_df.to_latex(float_format="%.3f"))
    with open(f"results_{model_name.replace('/', '_')}_{dataset_name}.csv", "w") as f:
         writer = csv.writer(f)
         writer.writerow(["name", "result"])
         for name, result in baseline_results.items():
              writer.writerow([name, result])
              if isinstance(result, (int, float)):
                   print(f"{name}: {result:.3f}")
              else:
                   print(f"{name}: {result}")

    # Log to wandb
    if wandb_run is not None:
         wandb_run.log(baseline_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
         "--model-name",
         type=str,
         help="Huggingface Hub identifier for llm.",
         default=LLM_MODEL
    )
    parser.add_argument(
         "--baseline-methods",
         nargs="+",
         choices=BASELINE_METHODS,
         default=BASELINE_METHODS
    )
    parser.add_argument(
         "--dataset-name",
         type=str,
         help="Name of the dataset.",
         choices=DATASETS
    )
    parser.add_argument(
         "--num-in-context-samples",
         type=int,
         default=NUM_IN_CONTEXT_SAMPLES
    )
    parser.add_argument(
        "--input-parts",
        nargs="+",
        type=str,
        default=INPUT_PARTS
    )
    parser.add_argument(
         "--calibration-batch-size",
         type=int,
         default=CALIBRATION_BATCH_SIZE,
    )
    parser.add_argument(
         "--calibration-learning-rate",
         type=float,
         default=CALIBRATION_LEARNING_RATE
    )
    parser.add_argument(
        "--calibration-num-steps", 
        type=int, 
        default=CALIBRATION_NUM_STEPS
    )
    parser.add_argument(
         "--calibration-valid-interval",
         type=int,
         default=CALIBRATION_VALID_INTERVAL
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device run the experiment on llm.",
    )
    parser.add_argument(
        "--data-dir",
        type=str, 
        default=DATA_DIR, 
        help="Directory containing data."
    )
    parser.add_argument(
        "--img-dir", 
        type=str, 
        help="Directory to plot images into.", 
        default=IMG_DIR
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULT_DIR,
        help="Directory containing result.",
    )
    parser.add_argument(
        "--use-binary-targets",
        action="store_true",
        default=True,
        help="Whether to use binary targets instead of clustering targets as in Apricot.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=SEED, 
        help="Random seed used for experiments."
    )
    parser.add_argument(
        "--track-emissions",
        action="store_true",
        default=False,
        help="Whether to track CO2eq emissions produced during the experiments.",
    )
    parser.add_argument(
        "--knock",
        action="store_true",
        default=False,
        help="Whether to announce experimental results via knockknock.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Whether to track experiments via Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Name of the run in Weights & Biases.",
    )
    args = parser.parse_args()

    # ACCELERATION
    device = args.device
    try:
        if torch.backends.mps.is_available():
            device = torch.device(device)
        elif torch.cuda.is_available():
            device = torch.device(device)
            torch.backends.cudnn.benchmark = True
    except Exception as e:
        warnings.warn(f"Error setting device: {str(e)}")

    tracker = None
    wandb_run = None
    timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
    #add 
    if not args.wandb_name:
        args.wandb_name = timestamp + "_" + args.model_name + "_"  + args.dataset_name

    if args.wandb:
         wandb_run = wandb.init(
              project=PROJECT_NAME,
              dir=os.path.join(args.data_dir, "wandb"),
              tags=[args.dataset_name, args.model_name],
              settings=wandb.Settings(start_method="fork"),
              config={
                   "model_name": args.model_name,
                   "baseline_methods": args.baseline_methods,
                   "dataset_name": args.dataset_name,
                   "num_in_context_samples": args.num_in_context_samples,
                   "timestamp": timestamp
              },
              name=args.wandb_name
         )


    #add knockknock
    if args.knock:
         if not SECRET_IMPORTED:
              raise ImportError(
                   "secret.py wasn't found, please create the file or add environment variables"
              )
         
         run_single_suffix_experiment = telegram_sender(
              token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
              )(run_calibration_benchmark)

    #add CO2 emission tracker
    
    # if args.track_emissions:
    #     timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
    #     emission_path = os.path.join(EMISSION_DIR, timestamp)
    #     os.makedirs(emission_path)
    #     tracker = OfflineEmissionsTracker(
    #         project_name=PROJECT_NAME,
    #     )

    run_calibration_benchmark(
         model_name=args.model_name,
         baseline_methods=args.baseline_methods,
         dataset_name=args.dataset_name,
         num_in_context_samples=args.num_in_context_samples,
         use_binary_targets=args.use_binary_targets,
         input_parts=args.input_parts,
         device=device,
         data_dir=args.data_dir,
         calibration_batch_size=args.calibration_batch_size,
         calibration_learning_rate=args.calibration_learning_rate,
         calibration_num_steps=args.calibration_num_steps,
         calibration_valid_interval=args.calibration_valid_interval,
         img_dir=args.img_dir,
         result_dir=args.results_dir,
         seed=args.seed,
         wandb_run=wandb_run
    )

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
