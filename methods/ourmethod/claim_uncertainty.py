"""
Module for extracting claims from QA dataset answers and measuring their uncertainty.
"""
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from tqdm import tqdm
from methods.ourmethod.claim_extractor import ClaimExtractor, Claim
# We format as context + claim and get logprobs for the claim portion
import tiktoken
from src.prompts import(
    SLOW_THINKING_PROMPT,
    QUAL_VERBALIZED_CONF_PROMPT,
    MEAN_DECISIVENESS_VERBALIZED_CONF_PROMPT,
)
from src.evaluation import extract_verbalized_confidence
from src.constant_vals import (
    QUALITATIVE_SCALE,
    MEAN_DECISIVENESS_SCALE,
    CLOSED_BOX_MODELS,
    MAX_INPUT_LENGTH,
)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ClaimWithUncertainty:
    """Class representing a claim with its uncertainty score."""
    text: str
    uncertainty: float   # combined confidence: alpha * freq + (1-alpha) * ver
    num_tokens: int
    ver_confidence: float = 0.0   # verification-only confidence (ver signal)
    freq_confidence: float = 0.0  # consistency-based confidence (freq signal)
    is_supported: bool = False
    error_type: Optional[str] = None


class ClaimUncertaintyMeasurer:
    """Extracts claims from QA answers and measures uncertainty for each claim."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        claim_extractor_model: Optional[str] = None,
    ):
        """Initialize claim uncertainty measurer.
        
        Args:
            model_name: Name of the model to use for computing log probabilities
            device: Device to run the model on (auto-detects if None: cuda > mps > cpu)
            claim_extractor_model: Model name for claim extraction (defaults to model_name)
        """
        self.model_name = model_name
        
        self.device = device
        self.claim_extractor_model = claim_extractor_model
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Initialize claim extractor
        self.claim_extractor = ClaimExtractor(
            claim_extractor_model=self.claim_extractor_model,
            device=device
        )
        
        # Initialize model and tokenizer for uncertainty computation
        if self.model_name not in CLOSED_BOX_MODELS:
            # 4-bit quantization for Apple Silicon
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.float16
            # )
            # Load model configuration
            config = AutoConfig.from_pretrained(
                self.model_name,
            )
            config.max_length = 512  # Set max length in config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                # quantization_config=quantization_config,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # device_map="auto",
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding='max_length', padding_side="left", max_length=MAX_INPUT_LENGTH)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
            self.model.eval()

    def compute_claim_confidence(
        self,
        claim: str,
        question: Optional[str] = None,
        openai_client: Optional[OpenAI] = None
    ) -> float:
        """Compute confidence of a claim.
        
        Args:
            claim: The claim text to compute confidence for
            question: Optional question context to prepend to the claim
            openai_client: Optional OpenAI client for GPT models
            
        Returns:
            Confidence of the claim
        """
        if self.model_name in CLOSED_BOX_MODELS:
            try:
                # Evaluate probability of claim tokens given context
                # This function gets the tokeniser for the specific model, especially in OpenAI API.
                if "gpt" in self.model_name.lower():
                    encoding = tiktoken.encoding_for_model(self.model_name)
                else:
                    encoding = tiktoken.get_encoding("o200k_base")

                if question:
                    # Format: context + claim
                    eval_prompt = f"{question}\n{claim}"
                    eval_prompt_verbalized = f"{question}\n{claim} {MEAN_DECISIVENESS_VERBALIZED_CONF_PROMPT}"
                    question_text = question
                else:
                    eval_prompt = f"{claim}"
                    question_text = ""

                # TODO: when no logprobs extract verbalized confidence from eval_response_verbalized or combine with self-evaluation 
                eval_response_verbalized = openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SLOW_THINKING_PROMPT},
                        {"role": "user", "content": eval_prompt_verbalized}
                    ],
                    max_completion_tokens=512,
                )
                confidence = eval_response_verbalized.choices[0].message.content
                confidence, successful = extract_verbalized_confidence([confidence], mode="qualitative", expression_mapping=MEAN_DECISIVENESS_SCALE)
                return float(confidence[0]) # return the first and only confidence value
            except Exception as e:
                print(f"Error computing verbalized confidence with OpenAI: {e}")
                return np.exp(np.log(1e-10))  # Return very low log prob on error
        else:
            # Use local model - compute probability of claim given context
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer must be initialized for non-GPT models")
            
            # Prepare input: context + claim
            if question:
                # Format: context + claim (we'll compute log prob of claim tokens)
                full_text = f"{question}\n{claim}"
                full_text_verbalized = f"{question}\n{claim} {MEAN_DECISIVENESS_VERBALIZED_CONF_PROMPT}"
            else:
                full_text = claim
            
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            inputs_verbalized = self.tokenizer(
                full_text_verbalized,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Tokenize separately to find claim start position
            if question:
                question_inputs = self.tokenizer(
                    question,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                claim_start_idx = question_inputs["input_ids"].shape[1]
            else:
                claim_start_idx = 0
            
            with torch.no_grad():
                # TODO: add verbalized confidence from self-evaluation
                outputs = self.model(**inputs)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                
                # Get log probabilities
                log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
                
                # Extract log probs for actual tokens in the claim
                input_ids = inputs["input_ids"]  # [batch_size, seq_len]
                seq_len = input_ids.shape[1]
                
                # Gather log probs for each token in the claim
                # Note: log_probs[i] predicts token at position i+1
                # So for token at position i, we use log_probs[i-1]
                token_log_probs = []
                for i in range(claim_start_idx, seq_len):
                    token_id = input_ids[0, i].item()
                    # Skip special tokens
                    if token_id in self.tokenizer.all_special_ids:
                        continue
                    # Get log prob of this token (log_probs[i-1] predicts token at position i)
                    if i > 0:
                        token_log_prob = log_probs[0, i - 1, token_id].item()
                        token_log_probs.append(token_log_prob)
                
                if len(token_log_probs) == 0:
                    outputs_verbalized = self.model(**inputs_verbalized)
                    generated_answer_ids = outputs_verbalized["sequences"][:, inputs_verbalized.shape[1] :].squeeze(0)
                    verbalized_confidence = self.tokenizer.batch_decode(generated_answer_ids, skip_special_tokens=True)
                    confidence, successful = extract_verbalized_confidence([verbalized_confidence], mode="qualitative", expression_mapping=MEAN_DECISIVENESS_SCALE)
                    return float(confidence[0]) # return the first and only confidence value
                
                avg_log_prob = np.mean(token_log_probs)
                return np.exp(float(avg_log_prob)) # uncertainty is the negative of log probability, higher log probability = lower uncertainty    

    def sample_n_responses(
        self,
        question: str,
        n: int = 5,
        temperature: float = 0.7,
        openai_client: Optional[OpenAI] = None,
    ) -> List[str]:
        """Sample N diverse responses to a question for consistency estimation.

        Args:
            question: The question to answer
            n: Number of responses to sample
            temperature: Sampling temperature (>0 for diversity)
            openai_client: OpenAI client (required for closed-box models)

        Returns:
            List of N sampled response strings
        """
        responses = []
        if self.model_name in CLOSED_BOX_MODELS:
            if openai_client is None:
                raise ValueError("openai_client required for closed-box model sampling")
            for _ in range(n):
                try:
                    resp = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": SLOW_THINKING_PROMPT},
                            {"role": "user", "content": question},
                        ],
                        temperature=temperature,
                        max_completion_tokens=512,
                    )
                    responses.append(resp.choices[0].message.content or "")
                except Exception as e:
                    print(f"Error sampling response: {e}")
                    responses.append("")
        else:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model and tokenizer required for open-box sampling")
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                for _ in range(n):
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    response = self.tokenizer.decode(
                        output_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    responses.append(response)
        return responses

    def compute_claim_freq(
        self,
        claim: str,
        sampled_responses: List[str],
        question: Optional[str] = None,
        openai_client: Optional[OpenAI] = None,
        verifier_model: str = "gpt-4o-mini",
    ) -> float:
        """Compute freq(c_i) = fraction of sampled responses that verifier-confirm the claim.

        For each sampled response, a lightweight GPT-4o-mini verifier checks whether
        the response explicitly supports the claim. The returned freq is the fraction
        of responses where the verifier answers 'Yes'.

        Args:
            claim: The atomic claim to check
            sampled_responses: N sampled responses to the same question
            question: Question context (prepended to the verification prompt)
            openai_client: OpenAI client used for the verifier call
            verifier_model: Model used as verifier (default: gpt-4o-mini)

        Returns:
            Fraction in [0, 1] of sampled responses that support the claim
        """
        if not sampled_responses:
            return 0.0
        if openai_client is None:
            return 0.0

        verify_template = (
            "Question: {question}\n"
            "Response: {response}\n"
            "Claim: {claim}\n\n"
            "Does the response explicitly support or confirm this claim? "
            "Answer with only 'Yes' or 'No'."
        )
        confirmed = 0
        for response in sampled_responses:
            if not response.strip():
                continue
            prompt = verify_template.format(
                question=question or "",
                response=response,
                claim=claim,
            )
            try:
                result = openai_client.chat.completions.create(
                    model=verifier_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=5,
                    temperature=0.0,
                )
                answer = result.choices[0].message.content.strip().lower()
                if answer.startswith("yes"):
                    confirmed += 1
            except Exception as e:
                print(f"Error in consistency verification: {e}")
        return confirmed / len(sampled_responses)

    def measure_uncertainty_for_answer(
        self,
        answer: str,
        question: Optional[str] = None,
        openai_client: Optional[OpenAI] = None,
        alpha: float = 0.0,
        n_consistency_samples: int = 0,
        verifier_model: str = "gpt-4o-mini",
    ) -> List[ClaimWithUncertainty]:
        """Extract claims from an answer and measure uncertainty for each.

        Args:
            answer: The answer text to extract claims from
            question: Optional question context
            openai_client: Optional OpenAI client for GPT models
            alpha: Weight for consistency signal; combined = alpha*freq + (1-alpha)*ver.
                   Default 0.0 reproduces the verification-only baseline.
            n_consistency_samples: Number of responses to sample for freq(c_i).
                   0 skips consistency computation (default, current behaviour).
            verifier_model: Cheap verifier model for consistency checks.

        Returns:
            List of ClaimWithUncertainty with ver_confidence, freq_confidence, and
            combined uncertainty filled in.
        """
        # Extract claims from answer
        try:
            claims = self.claim_extractor.extract_claims(answer, openai_client)
        except Exception as e:
            print(f"Error extracting claims: {e}")
            return []

        if len(claims) == 0:
            return []

        # Sample N responses once (shared across all claims for this answer)
        sampled_responses: List[str] = []
        if n_consistency_samples > 0 and question is not None:
            try:
                sampled_responses = self.sample_n_responses(
                    question=question,
                    n=n_consistency_samples,
                    openai_client=openai_client,
                )
            except Exception as e:
                print(f"Error sampling responses for consistency: {e}")

        # Compute uncertainty for each claim
        claims_with_uncertainty = []
        for claim in claims:
            # --- verification signal (ver) ---
            ver_conf = self.compute_claim_confidence(
                claim.text,
                question=question,
                openai_client=openai_client,
            )

            # --- consistency signal (freq) ---
            freq_conf = 0.0
            if sampled_responses:
                freq_conf = self.compute_claim_freq(
                    claim=claim.text,
                    sampled_responses=sampled_responses,
                    question=question,
                    openai_client=openai_client,
                    verifier_model=verifier_model,
                )

            # --- combined confidence ---
            combined = alpha * freq_conf + (1.0 - alpha) * ver_conf

            # Count tokens (rough estimate)
            if self.tokenizer:
                num_tokens = len(self.tokenizer.encode(claim.text))
            else:
                num_tokens = int(len(claim.text.split()) * 1.3)

            claims_with_uncertainty.append(
                ClaimWithUncertainty(
                    text=claim.text,
                    uncertainty=combined,
                    num_tokens=num_tokens,
                    ver_confidence=ver_conf,
                    freq_confidence=freq_conf,
                )
            )

        return claims_with_uncertainty

    def get_aggregated_uncertainty(
        self,
        aggregation: str = "mean",
        claims_with_uncertainty: Optional[List[ClaimWithUncertainty]] = None,
    ) -> float:
        """Get overall uncertainty score for an answer based on its claims.
        
        Args:
            aggregation: How to aggregate claim uncertainties ("mean", "min", "max", "weighted")
            claims_with_uncertainty: Optional list of claims with their uncertainty scores
            
        Returns:
            Overall uncertainty score (lower is more uncertain, higher is more certain)
        """
        
        if len(claims_with_uncertainty) == 0:
            return np.exp(np.log(1e-10)) # Very uncertain if no claims extracted
        
        uncertainties = [c.uncertainty for c in claims_with_uncertainty]
        
        if aggregation == "mean":
            return np.mean(uncertainties)
        elif aggregation == "min":
            return np.min(uncertainties)
        elif aggregation == "max":
            return np.max(uncertainties)
        elif aggregation == "weighted":
            # Weight by number of tokens
            weights = [c.num_tokens for c in claims_with_uncertainty]
            total_weight = sum(weights)
            if total_weight == 0:
                return np.mean(uncertainties)
            return np.average(uncertainties, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
