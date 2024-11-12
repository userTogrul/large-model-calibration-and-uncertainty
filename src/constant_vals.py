"""
    A module to define default values and configurations
"""

# Default paths
PROJECT_NAME = "llm_calibration"
HF_HOME = '/data1/toghrul/cache/huggingface'
HF_CACHE_DIR = '/data1/toghrul/cache'
DATA_DIR = "./data"
RESULT_DIR = "./results"
IMG_DIR = "./src/img"
EMISSION_DIR = "./emissions"

# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4o-mini"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct" 
CALIBRATION_MODEL = "microsoft/deberta-v3-base"
SENTENCE_EMBEDDING_MODEL = "all-mpnet-base-v2"
BLACK_BOX_MODELS = ["gpt-4", "gpt-4o-mini"]
WHITE_BOX_MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]

# Calibrator hyperparameters
CALIBRATION_BATCH_SIZE = 64
CALIBRATION_LEARNING_RATE = 0.01
CALIBRATION_NUM_STEPS = 200
CALIBRATION_VALID_INTERVAL = 20
MAX_INPUT_LENGTH = 512 # hyperparameter
NUM_IN_CONTEXT_SAMPLES = 10
END_OF_GENERATION_TOKENS = [
    "Question:",
    " Question:",
    "Question: ",
    "\n",
    "Answer:",
    "\nQuestion:",
    " Answer:",
    "Q:",
]
# TODO
# can we split into 10 parts? As number of confidence bins
QUALITATIVE_SCALE = {
    "Very low": 0,
    "Low": 0.3,
    "Somewhat low": 0.45,
    "Medium": 0.5,
    "Somewhat high": 0.65,
    "High": 0.7,
    "Very high": 1,
}
SEED = 1212
INPUT_PARTS= ["question", "answer"] # Default composition of inputs
END_OF_GENERATION_TOKENS = [
    "Question:",
    " Question:",
    "Question: ",
    "\n",
    "Answer:",
    "\nQuestion:",
    " Answer:",
    "Q:",
]
ALLOWED_INPUTS = {
    "question",
    "answer",
    "cot_answer",
    "qualitative",
    "quantitative",
}  # Allowed parts for input
CALIBRATION_PARAMS = {
    "learning_rate": CALIBRATION_LEARNING_RATE,
    "num_training_steps": CALIBRATION_NUM_STEPS,
    "eval_interval": CALIBRATION_VALID_INTERVAL,
}

# dataset details
DATASETS = ("trivia_qa", "truthful_qa")
DATASET_SPLIT_SIZES = {
    "trivia_qa": {"train": 12000, "test": 1500},
    "truthful_qa": {"train": 409, "test": 408}
}

# BASELINES & METRICS
BASELINE_METHODS = [
    "seq_likelihood",
    "cot_seq_likelihood",
    "cot_qual_verbalized_uncertainty",
    "ps_seq_likelihood",
    "ts_seq_likelihood",
    # "sb_seq_likelihood",
]
EVAL_METRIC_ORDER = ["brier_score", "ece", "smece", "bleu", "auroc"]
