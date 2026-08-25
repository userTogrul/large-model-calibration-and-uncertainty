"""
Module for classifying claims based on reference context.
"""
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .claim_extractor import Claim
from openai import OpenAI

class ClaimClassifier:
    """Classifies claims against reference context using CoT reasoning."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize claim classifier.
        
        Args:
            model_name: Name of the model to use for classification
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        
        # Classification labels
        self.labels = [
            "supported",
            "contradicted",
            "absent",
            "partially supported",
            "unevaluatable",
        ]
        
        # Error type labels
        self.error_types = [
            "number",
            "entity",
            "false-concat",
            "attribution-failure",
            "overgeneralization",
            "reasoning-error",
            "hyperbole",
            "temporal",
            "context-based-meaning",
            "other"
        ]
        
        # TODO: Make the prompt like in HalluMeasure W/CoT
        # Prompt templates
        self.single_claim_prompt = """
        Evaluate if the following claim is supported by the reference context.
        Think step by step about how the claim relates to the context.
        
        Context: {context}
        
        Claim: {claim}
        
        Step-by-step evaluation:
        1) First, let's understand what the claim is stating:
        {claim_analysis}
        
        2) Now, let's check how this relates to the context:
        {context_analysis}
        
        3) Based on this analysis, the claim is:
        {label}
        
        4) If the claim is not fully supported, the type of error is:
        {error_type}
        """
        
        self.batch_claim_prompt = """
        <task>
        You will act as an expert annotator to evaluate claims against a provided source text.
        The source text will be given within <source></source> XML tags.
        The claims to evaluate will be provided within <claims></claims> XML tags.
        For each claim, follow these steps:

        Step 1: Read and fully understand the claim. It is a short, standalone sentence containing a single piece of information related to the source text.
        Step 2: Thoroughly analyze how the claim relates to the information in the source text. Then, write your reasoning in 1-3 sentences to determine the most appropriate label to describe the claim's truthfulness based on the source text. The definitions of the labels are provided below.
        
        1. supported: The claim is definitely true according to the source text.
        2. contradicted: The claim is definitely false according to the source text.
        3. absent: The source text provides no information to confirm or deny the claim.
        4. partially supported: The claim is mostly consistent with the source text, but some words are added, omitted, or inaccurate. This is a 'near miss' label. If either '2-contradicted' or '3-absent' seems like an appropriate label, use these instead. Note that claims that should be attributed to a particular source (e.g. quotes) are '4-partially supported' if the attribution is missing.
        5. unevaluatable: The claim cannot be interpreted as a statement to evaluate against the source text (e.g. it is a question or instruction).

        Step 3: Write the label for the claim based on your reasoning in Step 2.
        Step 4: If the label in Step 3 is '2-contradicted', '3-absent', or '4-partially supported', then thoroughly analyze the specific error or inaccuracy present in the claim based on provided source text. Then, provide your reasoning in 1-3 sentences to determine the error. Given below are the definitions of those specific error types (i.e., sublabels). However, if the label in Step 3 is '1-supported', simply write 'None - claim is supported', and set the sublabel to 'None'.

        1. number: the claim has a different number than the original context (e.g., 50m vs. 60m).
        2. entity: the claim includes swapped or incorrectly selected entities (e.g., one named entity vs. another).
        3. false-concat: the claim inappropriately combines information about two or more different entities or events.
        4. attribution-failure: the claim does not correctly attribute the information it contains, e.g. failing to attribute quoted material to the correct person or entity named in the article.
        5. overgeneralization: the claim is based on accurate contextual information but it is too broad or too general to be supported by the context.
        6. reasoning-error: the claim is based on accurate contextual information but includes a reasoning error which makes the claim inaccurate.
        7. hyperbole: the claim is based on accurate contextual information but is inappropriately strengthened or overstated.
        8. temporal: the claim does not accurately incorporate tense, modality (e.g. might vs. will) or time reference in relation to the context.
        9. context-based-meaning: the claim includes incorrect interpretation of idiomatic language, homonyms, or words with multiple meanings, therefore failing to capture the intended meaning.
        10. other: all other types of errors are captured in this category.

        Step 5: Write the sublabel based on your reasoning in Step 4.

        Repeat these steps for each claim provided.

        You will output the list of dictionaries. Each dictionary will have below format:
        {{
        "claim": "<CLAIM TEXT>",
        "thought_for_label": "<THOUGHT FOR LABEL FOR THE CLAIM>",
        "label": "CLAIM LABEL",
        "thought_for_sublabel": "<THOUGHT FOR SUBLABEL FOR THE CLAIM>",
        "sublabel": "CLAIM SUBLABEL"
        }}
        </task>
        
        <source>: 
        {context}
        </source>
        \n
        <claims>
        {claims}
        </claims>
        """
            
        if "gpt" not in model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def classify_claim(
        self,
        claim: Claim,
        context: str,
        openai_client: Optional[OpenAI] = None
    ) -> Tuple[str, Optional[str]]:
        """Classify a single claim against the reference context in one inference time.
        
        Args:
            claim: Claim to classify
            context: Reference context
            openai_client: Optional OpenAI client instance. If not provided, will create one.
            
        Returns:
            Tuple of (classification label, error type if applicable)
        """
        # Format prompt
        prompt = self.single_claim_prompt.format(
            context=context,
            claim=claim.text,
            claim_analysis="[Analysis of claim meaning]",
            context_analysis="[Analysis of context support]",
            label="[Classification]",
            error_type="[Error type if applicable]"
        )
        
        if "gpt" in self.model_name.lower():
            # Use OpenAI API
            if openai_client is None:
                openai_client = OpenAI()
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates claims against reference context."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
        else:
            # Use local model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Parse classification and error type from response
        label = None
        error_type = None
        
        for line in result.split("\n"):
            if "3) Based on this analysis, the claim is:" in line:
                for l in self.labels:
                    if l in line.lower():
                        label = l
                        break
            elif "4) If the claim is not fully supported, the type of error is:" in line:
                for e in self.error_types:
                    if e in line.lower():
                        error_type = e
                        break
                        
        return label or "unevaluatable", error_type

    def classify_claims(
        self,
        claims: List[Claim],
        context: str,
        openai_client: Optional[OpenAI] = None
    ) -> List[Tuple[str, Optional[str]]]:
        """Classify multiple claims against the reference context in one inference time.
        
        Args:
            claims: List of claims to classify
            context: Reference context
            openai_client: Optional OpenAI client instance. If not provided, will create one.
            
        Returns:
            List of (classification label, error type) tuples
        """
        # Format batch prompt
        claims_text = "\n".join(f"{c.text}" for c in claims)
        
        prompt = self.batch_claim_prompt.format(
            context=context,
            claims=claims_text,
        )
        
        if "gpt" in self.model_name.lower():
            # Use OpenAI API
            if openai_client is None:
                openai_client = OpenAI()
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates claims against reference context."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
        else:
            # Use local model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True
            )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Parse results for each claim
        results = []
        current_result = {"claim": "", "label": None, "sublabel": None}
        
        for line in result.split("\n"):
            line = line.lower().strip()
            
            # Look for label in the line
            if '"label":' in line:
                label_part = line.split('"label":')[1].strip(' ",')
                for l in self.labels:
                    if l in label_part:
                        current_result["label"] = l
                        break
                        
            # Look for sublabel/error type in the line
            if '"sublabel":' in line:
                sublabel_part = line.split('"sublabel":')[1].strip(' ",')
                if "none" in sublabel_part or "supported" in sublabel_part:
                    current_result["sublabel"] = None
                else:
                    for e in self.error_types:
                        if e in sublabel_part:
                            current_result["sublabel"] = e
                            break
                
                # When we find a sublabel, we've reached the end of a claim result
                results.append((current_result["label"] or "unevaluatable", current_result["sublabel"]))
                current_result = {"claim": "", "label": None, "sublabel": None}
        
        # Fill in any missing results
        while len(results) < len(claims):
            results.append(("unevaluatable", None))
            
        return results
    