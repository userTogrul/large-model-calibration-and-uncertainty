"""
Main module for HalluMeasure hallucination detection.
"""
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from openai import OpenAI
from .claim_extractor import ClaimExtractor, Claim
from .claim_classifier import ClaimClassifier

@dataclass
class HalluMeasureResult:
    """Results from HalluMeasure analysis."""
    claims: List[Claim]
    hallucination_score: float
    error_type_distribution: Dict[str, float]
    supported_claims: List[Claim]
    unsupported_claims: List[Claim]

class HalluMeasure:
    """Main class for hallucination measurement using chain-of-thought reasoning."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_classify: bool = False
    ):
        """Initialize HalluMeasure.
        
        Args:
            model_name: Name of the model to use
            device: Device to run models on
            batch_classify: Whether to classify claims in batches
        """
        self.claim_extractor = ClaimExtractor(model_name=model_name, device=device)
        self.claim_classifier = ClaimClassifier(
            model_name=model_name,
            device=device,
        )
        
    def measure_hallucination(
        self,
        response: str,
        context: str,
        openai_client: Optional[OpenAI] = None
    ) -> HalluMeasureResult:
        """Measure hallucination in an LLM response against reference context.
        
        Args:
            response: LLM-generated response to analyze
            context: Reference context to check claims against
            openai_client: Optional OpenAI client instance. If not provided, will create one.
            
        Returns:
            HalluMeasureResult containing analysis results
        """
        if openai_client is None:
            openai_client = OpenAI()
            
        # Extract claims
        claims = self.claim_extractor.extract_claims(response, openai_client)
        
        if not claims:
            return HalluMeasureResult(
                claims=[],
                hallucination_score=0.0,
                error_type_distribution={},
                supported_claims=[],
                unsupported_claims=[]
            )
            
        # Classify claims
        classifications = self.claim_classifier.classify_claims(claims, context, openai_client)
        
        # Process results
        supported_claims = []
        unsupported_claims = []
        error_types = {}
        
        for claim, (label, error_type) in zip(claims, classifications):
            claim.is_supported = (label == "supported")
            claim.error_type = error_type
            
            if claim.is_supported:
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
                if error_type:
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
        # Calculate hallucination score and error distribution
        hallucination_score = len(unsupported_claims) / len(claims)
        
        error_type_distribution = {}
        if error_types:
            total_errors = sum(error_types.values())
            error_type_distribution = {
                k: v/total_errors for k, v in error_types.items()
            }
            
        return HalluMeasureResult(
            claims=claims,
            hallucination_score=hallucination_score,
            error_type_distribution=error_type_distribution,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims
        )
        
    def batch_measure_hallucination(
        self,
        responses: List[str],
        contexts: Union[str, List[str]],
        openai_client: Optional[OpenAI] = None
    ) -> List[HalluMeasureResult]:
        """Measure hallucination in multiple responses.
        
        Args:
            responses: List of LLM-generated responses
            contexts: Either a single context or list of contexts matching responses
            openai_client: Optional OpenAI client instance. If not provided, will create one.
            
        Returns:
            List of HalluMeasureResult for each response
        """
        if isinstance(contexts, str):
            contexts = [contexts] * len(responses)
            
        if len(contexts) != len(responses):
            raise ValueError("Number of contexts must match number of responses")
            
        if openai_client is None:
            openai_client = OpenAI()
            
        return [
            self.measure_hallucination(response, context, openai_client)
            for response, context in zip(responses, contexts)
        ] 