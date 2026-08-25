"""
    Self-correction prompts
    Source:
        https://github.com/yuanyige/fact-calibration.git 
"""

AGG_FACT_FORMAT='''
You will be given a SENTENCE along with its suggested CORRECTION. 
Your task is to apply the CORRECTION to the SENTENCE as indicated and return the modified sentence under REVISE.


Example 1:

SENTENCE:
Fei-Fei Li is a female historian.

CORRECTION:
a female historian -> a female scientist.

REVISE:
Fei-Fei Li is a female scientist.


Example 2:

SENTENCE
Quoc Le joined ByteDance to lead its AI research team.

CORRECTION:
ByteDance -> Google

REVISE:
Quoc Le joined Google to lead its AI research team.


Your Task:

SENTENCE:
{ori_fact}

CORRECTION:
{revise_fact}
'''



GPT4_EVAL_FORMAT="""
You will be provided with a QUESTION, its RESPONSE, and all facts extracted from the RESPONSE under the heading "ALL FACTS".

You will also be provided with a specific fact under the heading "TARGET FACT 1", which is included in ALL FACTS. 
Additionally, you will be given a modified version of this target fact under the heading "TARGET FACT 2".
Based on your knowledge, evaluate whether the modification of the target fact is an improvement, the same, or a regression.

An improvement implies:
1. More accurate information,
2. Greater relevance to the question,
3. Minimal overlap with other facts in ALL FACTS.

A regression implies:
1. Introduction of erroneous or inaccurate information,
2。 Lower relevance to the question,
3. Repetition or introduction of information that is already provided with other facts in ALL FACTS.

Special attention should be paid to ensuring that a good modification does not overly repeat or directly copy information from other facts in ALL FACTS.

QUESTION:
{question}

RESPONSE:
{response}

ALL FACTS:
{all_facts}

TARGET FACT 1:
{ori_fact}

TARGET FACT 2:
{revise_fact}

First, provide a one-sentence comparison of the two facts and explain whether you think the modification is an improvement, the same, or a regression.
Second, on a new line, state only "IMPROVED", "SAME", or "REGRESSED" to indicate the effectiveness of the modification. Your response should use the following format:

COMPARISON:
<one-sentence comparison and explanation>

REVISION: <"IMPROVED", "SAME", or "REGRESSED">
"""
