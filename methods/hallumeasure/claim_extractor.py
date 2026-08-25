"""
Module for extracting atomic claims from LLM responses.
"""
from typing import List, Optional
from dataclasses import dataclass
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

@dataclass
class Claim:
    """Class representing an atomic claim extracted from text."""
    text: str
    confidence: float = 0.0
    is_supported: bool = False
    error_type: Optional[str] = None

class ClaimExtractor:
    """Extracts atomic claims from LLM responses using a claim extraction model."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize claim extractor.
        
        Args:
            model_name: Name of the model to use for claim extraction
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        
        # Prompt template for claim extraction
        self.extraction_prompt = """
        A claim is a short sentence containing a single piece of information. 
        You will extract claims from a given text inside <text></text> XML tags.
        
        Here are the  "Task-rules" you must follow when generating claims.
        <task-rules>
            <rule>The claim should be entriely self-contained. For instance, the claim should be comprehended without relying on other claims.</rule>
            <rule>The claim should not contain pronouns. If there are pronouns in the input text, replace them with their corresponding nouns when generating the claims.</rule>
            <rule>The claim should not exceed 15 words.</rule>
            <rule>You will always output a list of the extracted claims.</rule>
            <rule>You will always change double quotes to single quotes in the claims. For example, write 'glass' instead of "glass".</rule>
        </task-rules>

        <text>:
        {text}
        </text>
        \n\nAssistant:
        """
        
        if "gpt" not in model_name.lower():
            # 4-bit quantization for Apple Silicon
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, low_cpu_mem_usage=True, device_map="auto").to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_claims(self, text: str, openai_client: Optional[OpenAI] = None) -> List[Claim]:
        """Extract atomic claims from input text.
        
        Args:
            text: Input text to extract claims from
            openai_client: Optional OpenAI client instance. If not provided, will create one.
            
        Returns:
            List of extracted claims
        """
        # Format prompt
        prompt = self.extraction_prompt.format(text=text)
        
        if "gpt" in self.model_name.lower():
            # Use OpenAI API
            if openai_client is None:
                openai_client = OpenAI()
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts atomic claims from text."},    
                    {"role": "user", "content": prompt}
                ]
            )
            claims_text = response.choices[0].message.content

            if claims_text is None:
                return []
        else:
            # Use local model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            claims_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse claims from response
        claims = []
        for line in claims_text.strip().split("\n"):
            if line and not line.isspace():
                claims.append(Claim(text=line.strip()))
                
        return claims 