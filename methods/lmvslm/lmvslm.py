"""
Implementation of LMvsLM method for detecting factual errors via cross examination.
Based on the paper: LM vs LM: Detecting Factual Errors via Cross Examination
https://arxiv.org/pdf/2305.13281
"""
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time
from secret import OPENAI_API_KEY, BASE_URL
class LMvsLM:
    """
    LMvsLM class implements the cross-examination approach for detecting factual errors.
    """
    def __init__(
        self,
        examiner_model: str = "gpt-4o-mini",
        examinee_model: Union[str, AutoModelForCausalLM] = "gpt-4",
        max_questions: int = 5,
        device: str = "cpu",
    ):
        """
        Initialize LMvsLM.
        
        Args:
            examiner_model: The model to use as examiner (GPT model)
            examinee_model: The model to use as examinee (can be GPT model name or HF model instance)
            max_questions: Maximum number of questions in cross examination
            device: Device to run HF models on

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1")
            
        self.examiner_model = examiner_model
        self.examinee_model = examinee_model
        self.max_questions = max_questions
        self.device = device
        # Initialize HF model and tokenizer if examinee is open-access
        self.is_examinee_openai = isinstance(examinee_model, str) and ("gpt" in examinee_model.lower() or "deepseek" in examinee_model.lower())
        if not self.is_examinee_openai:
            if isinstance(examinee_model, str):
                try:
                    # Initialize tokenizer first to ensure compatibility
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        examinee_model,
                    )
                    # self.tokenizer.pad_token = self.tokenizer.bos_token  # Set padding token
                    
                    # Load model configuration
                    config = AutoConfig.from_pretrained(
                        examinee_model,
                    )
                    config.max_length = 512  # Set max length in config
                    
                    # Initialize model with proper configuration
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        examinee_model,
                        config=config,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    ).to(device)
                    
                    # Ensure pad token ID is set in model config
                    # self.hf_model.config.pad_token_id = self.tokenizer.bos_token_id
                        
                except Exception as e:
                    raise ValueError(f"Failed to load HuggingFace model and tokenizer: {str(e)}")
            else:
                # Use provided model instance
                self.hf_model = examinee_model
                self.hf_model = self.hf_model.to(device)

                # Initialize tokenizer from model's config
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_model.config._name_or_path,
                )

                # self.tokenizer.pad_token = self.tokenizer.bos_token  # Set padding token
                # self.hf_model.config.pad_token_id = self.tokenizer.bos_token_id
        
    def get_examinee_answer(
        self,
        question: str,
    ) -> str:
        """
        Get answer from examinee model, handling both GPT and HF models.
        
        Args:
            question: The question to ask
            client: OpenAI client instance (required for GPT models)
            
        Returns:
            str: The model's answer
            
        Raises:
            ValueError: If client is missing for GPT models or if the question is empty
            RuntimeError: If model generation fails
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
            
        if self.is_examinee_openai:
            client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=BASE_URL,
            )
            try:
                response = client.chat.completions.create(
                    model=self.examinee_model,
                    messages=[{"role": "user", "content": question}],
                )
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"Failed to get response from GPT model: {str(e)}")
        else:
            # Use HF model
            try:
                inputs = self.tokenizer(
                    question,
                    return_tensors="pt",
                ).to(self.device)
                
                with torch.no_grad() and torch.backends.cuda.sdp_kernel(
                    enable_flash=True # enable flash attention
                ):
                    outputs = self.hf_model.generate(
                        **inputs,
                        max_new_tokens=inputs["input_ids"].shape[1] + 20,
                    )
                
                # Decode and clean up the response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.strip()
                return response
            except Exception as e:
                raise RuntimeError(f"Failed to generate response from HuggingFace model: {str(e)}")
        
    def cross_examine(
        self,
        claim: str,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ) -> Tuple[float, List[Dict[str, str]]]:
        """
        Conduct cross examination of a claim.
        
        Args:
            claim: The claim to examine
            client: OpenAI client instance (required for examiner)
            base_url: Optional base URL for API
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            
        Returns:
            Tuple containing:
            - confidence score (1 - error probability)
            - list of QA interactions during examination
            
        Raises:
            ValueError: If claim is empty or client is None
            RuntimeError: If examination fails after max retries
        """
        # Initial prompt for examiner
        examiner_prompt = (
            "You are an examiner tasked with determining if the following claim is factually correct "
            "through cross-examination. Ask questions to verify the claim's accuracy.\n\n"
            f"Claim: {claim}\n\n"
            "Ask a question to verify this claim:"
        )
        
        interactions = []
        contradictions = 0
        total_questions = 0
        
        try:
            # Cross examination loop
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=BASE_URL,
            )
            is_no_questions = False
            for i in range(self.max_questions):
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Get question from examiner
                        examiner_response = client.chat.completions.create(
                            model=self.examiner_model,
                            messages=[{"role": "user", "content": examiner_prompt}],
                        )
                        question = examiner_response.choices[0].message.content
                        
                        # Get answer from examinee
                        answer = self.get_examinee_answer(question)
                        
                        # Record interaction
                        interactions.append({
                            "question": question,
                            "answer": answer
                        })
                        
                        # Have examiner evaluate answer
                        evaluation_prompt = (
                            f"Original claim: {claim}\n"
                            f"Question: {question}\n"
                            f"Answer: {answer}\n\n"
                            "Does this answer contradict or cast doubt on the original claim? "
                            "Respond with 'Yes' or 'No' and explain why."
                        )
                        
                        evaluation = client.chat.completions.create(
                            model=self.examiner_model,
                            messages=[{"role": "user", "content": evaluation_prompt}],
                        )
                        eval_response = evaluation.choices[0].message.content

                        if (eval_response is not None):
                            if "yes" in eval_response.lower():
                                contradictions += 1
                            elif "no" in eval_response.lower():
                                is_no_questions = True
                        else:
                            is_no_questions = True
                        total_questions += 1
                        break  # Success, exit retry loop
                    
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise RuntimeError(f"Failed to complete examination after {retry_count} retries: {str(e)}")
                        time.sleep(retry_delay)
                
                if is_no_questions:
                    break    
                # Update examiner prompt with context
                examiner_prompt = (
                    f"Based on previous questions and answers about the claim: {claim}\n"
                    "Ask another relevant question to further verify its accuracy:"
                )
                time.sleep(0.1) # sleep for 0.1 seconds to avoid rate limiting
        except Exception as e:
            raise RuntimeError(f"Error during cross examination: {str(e)}")
            
        # Calculate confidence score
        if total_questions == 0:
            confidence = 0.0
        else:
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, 1.0 - (contradictions / total_questions)))
            
        return confidence, interactions

    def get_confidence(
        self,
        claim: str,
    ) -> float:
        """
        Get confidence score for a claim.
        
        Args:
            claim: The claim to examine
            client: OpenAI client instance
            base_url: Optional base URL for API
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence, _ = self.cross_examine(claim)
        return confidence
        
    def cleanup(self):
        """
        Explicitly clean up resources used by the model.
        This should be called when you're done using the model to free up memory.
        """
        if hasattr(self, 'hf_model'):
            # Only delete if we created the model ourselves
            if isinstance(self.examinee_model, str):
                try:
                    self.hf_model.cpu()  # Move to CPU first
                    del self.hf_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Warning: Failed to cleanup model: {str(e)}")
            
    def __del__(self):
        """Attempt cleanup when object is deleted, but prefer explicit cleanup()"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor 