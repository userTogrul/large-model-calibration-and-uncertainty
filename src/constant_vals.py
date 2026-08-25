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
GPT_MODEL = "gpt-4o"
LLM_MODEL = "meta-llama/Llama-3.2-1B-Instruct" # "Qwen/Qwen2.5-0.5B-Instruct" 
SENTENCE_EMBEDDING_MODEL = "all-mpnet-base-v2"
CLOSED_BOX_MODELS = [
    "gpt-4", 
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5-mini",
    "gpt-5.2",
    "o4-mini",
    "aiproxy/deepseek-reasoner",
    "Deepseek-R1",
]
OPEN_BOX_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct", 
    "Qwen/Qwen2.5-7B-Instruct", 
    "Qwen/Qwen3-4B-Instruct-2507",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistralai/Mistral-7B-Instruct-v0.3", 
    "meta-llama/Llama-3.1-70B-Instruct", 
    "Qwen/Qwen2.5-72B-Instruct",
]

# Calibrator hyperparameters
DATA_BATCH_SIZE = 4
CALIBRATION_BATCH_SIZE = 32
CALIBRATION_LEARNING_RATE = 1e-2
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
# ALT_QUALITATIVE_SCALE = {
#     "Impossible": 0,
#     "Very low": 0.125,
#     "Low": 0.25,
#     "Somewhat low": 0.375,
#     "Medium": 0.5,
#     "Somewhat high": 0.635,
#     "High": 0.76,
#     "Very high": 0.885,
#     "Almost certain": 1
# }
QUALITATIVE_SCALE = {
    "Very low": 0,
    "Low": 0.3,
    "Somewhat low": 0.45,
    "Medium": 0.5,
    "Somewhat high": 0.65,
    "High": 0.7,
    "Very high": 1,
}
MEAN_DECISIVENESS_SCALE = {
    "Almost No Chance": 0.03,
    "Highly Unlikely": 0.06,
    "Improbable": 0.12,
    "Little Chance": 0.14,
    "Chances are Slight": 0.15,
    "Unlikely": 0.20,
    "We Doubt": 0.23,
    "Probably Not": 0.33,
    "About Even": 0.55,
    "Better than Even": 0.64,
    "Likely": 0.71,
    "Very Good Chance": 0.75,
    "Highly likely": 0.90,
    "Almost Certain": 0.93,
}
SEED = 1111
INPUT_PARTS= ["question", "answer"] # Default composition of inputs
ALLOWED_INPUTS = {
    "question",
    "answer",
    "cot_answer",
    "qualitative",
}  # Allowed parts for input
CALIBRATION_PARAMS = {
    "learning_rate": CALIBRATION_LEARNING_RATE,
    "num_training_steps": CALIBRATION_NUM_STEPS,
    "eval_interval": CALIBRATION_VALID_INTERVAL,
}

# dataset details
DATASETS = ("trivia_qa", "truthful_qa", "natural_questions")
DATASET_SPLIT_SIZES = {
    "trivia_qa": {"train": 12000, "test": 1500},
    "truthful_qa": {"train": 409, "test": 408},
    "natural_questions": {"train": 500, "test": 500},  # From Natural Questions paper
}

# BASELINES & METRICS
BASELINE_METHODS = [
    "seq_likelihood",
    "cot_seq_likelihood",
    "cot_qual_verbalized_uncertainty",
    "qual_verbalized_uncertainty",
    "ps_seq_likelihood",
    "ts_seq_likelihood",
    "lmvslm",
    "hallumeasure",
    "ourmethod",
    # "sb_seq_likelihood", #TODO
]
EVAL_METRIC_ORDER = ["ece", "smece", "brier_score", "auroc", "aurc", "bleu"]
