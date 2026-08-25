"""
Module for extracting atomic claims from LLM responses using slow thinking.
"""
from dataclasses import dataclass
from typing import List, Optional
import torch
from openai import OpenAI
from src.prompts import SLOW_THINKING_PROMPT

@dataclass
class Claim:
    """Class representing an atomic claim extracted from text."""
    text: str

class ClaimExtractor:
    """Extracts atomic claims from LLM responses using slow thinking."""

    def __init__(
        self,
        claim_extractor_model: str = "gpt-4o-mini",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize claim extractor.
        
        Args:
            claim_extractor_model: Name of the model to use for claim extraction
            device: Device to run the model on
        """
        self.claim_extractor_model = claim_extractor_model
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

    def extract_claims(
        self, 
        text: str, 
        openai_client: Optional[OpenAI] = None
    ) -> List[Claim]:
        """
        Extract claims from text.
        
        Args:
            text: The text to extract claims from
            openai_client: The OpenAI client to use for the API call
            
        Returns:
            A list of claims
        """
        # Format prompt
        prompt = self.extraction_prompt.format(text=text)
        
        # Use OpenAI API
        response = openai_client.chat.completions.create(
            model=self.claim_extractor_model,
            messages=[
                {"role": "system", "content": SLOW_THINKING_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        claims_text = response.choices[0].message.content

        if claims_text is None:
            return []

        # Parse claims from response
        claims = []
        for line in claims_text.strip().split("\n"):
            if line and not line.isspace():
                claims.append(Claim(text=line.strip()))
                
        return claims
