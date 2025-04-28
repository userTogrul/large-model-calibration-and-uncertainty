# Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review

\
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![last commit](https://img.shields.io/github/last-commit/userTogrul/large-model-calibration-and-uncertainty?color=green)](https://img.shields.io/github/last-commit/userTogrul/large-model-calibration-and-uncertainty?color=green)
[![Paper: arXiv](https://img.shields.io/badge/Paper-arXiv-%23B31B1B)](https://arxiv.org/abs/2504.18346)
[![Made with Love](https://img.shields.io/badge/Made%20with-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

This reposity contains review list and source code for various methods for quantification of uncertainty and calibration in large language models.

## Setup

Please download and install conda for python environment, and setup necessessary libraries:

    conda create -n large-model-calibration-and-uncertainty python=3.10
    conda activate large-model-calibration-and-uncertainty
    python -m pip install -r requirements.txt

Then create a `secret.py` file with necessary api keys as described in `secret_template.py` in the same folder as `main.py`.

## Usage

Run the script and type 'Yes' or 'y' when prompted.

For **Llama-3.1-8B-Instruct**:

    sh run_llama.sh

For **Mistral-7B-Instruct-v0.3**:

    sh run_mistral.sh

For **Qwen2.5-7B-Instruct**:

    sh run_qwen.sh

For **gpt-4 & gpt-4o**:

    sh run_openai.sh

For **deepseek-r1**:

    sh run_deepseek_r1.sh

## Literature review list

### Uncertanty Quantification

#### Sampling-based

- **Monte Carlo Dropout (MCD)**
  - Dropout as a bayesian approximation: Representing model uncertainty in deep learning. _Gal, Yarin, and Zoubin Ghahramani._ International conference on machine learning. PMLR, 2016. [[paper](https://arxiv.org/pdf/1506.02142)][[code](https://github.com/yaringal/DropoutUncertaintyExps/blob/master/readme.md)]
  - On uncertainty calibration and selective generation in probabilistic neural summarization: A benchmark study. _Zablotskaia, Polina, et al._ arXiv preprint arXiv:2304.08653 (2023). [[paper](https://arxiv.org/pdf/2304.08653)]

- **Model Ensemble.** Quantifying uncertainty in foundation models via ensembles. _Sun, Meiqi, et al._ NeurIPS 2022 Workshop on Robustness in Sequence Modeling. 2022.[[paper](https://openreview.net/pdf?id=LpBlkATV24M)]

- **R-U-SURE.** Ru-sure? uncertainty-aware code suggestions by maximizing utility across random user intents. _Johnson, Daniel D., Daniel Tarlow, and Christian Walder_ arXiv preprint arXiv:2303.00732 (2023). [[paper](https://arxiv.org/abs/2303.00732)][[code](https://github.com/google-research/r_u_sure)]

- **Clarification Ensemble.** Decomposing uncertainty for large language models through input clarification ensembling. _Hou, Bairu, et al._ arXiv preprint arXiv:2311.08718 (2023). [[paper](https://arxiv.org/pdf/2311.08718)][[code](https://github.com/UCSB-NLP-Chang/llm_uncertainty)]

- **Sampling with Perturbation for UQ.** Spuq: Perturbation-based uncertainty quantification for large language models. _Gao, Xiang, et al._ arXiv preprint arXiv:2403.02509 (2024).[[paper](https://arxiv.org/pdf/2403.02509)]

#### Entropy-based
- **Predictive Entropy.** Language models (mostly) know what they know.  _Kadavath, Saurav, et al._ arXiv preprint arXiv:2207.05221 (2022).[[paper](https://arxiv.org/pdf/2207.05221)][[code](https://github.com/iinemo/lm-polygraph)]

- **Length-normalized PE.** Uncertainty estimation in autoregressive structured prediction. _Malinin, Andrey, and Mark Gales_ arXiv preprint arXiv:2002.07650 (2020). [[paper](https://arxiv.org/pdf/2002.07650)]

- **Semantic Entropy**
  - Detecting hallucinations in large language models using semantic entropy. _Farquhar, Sebastian, et al._ Nature 630.8017 (2024): 625-630 [[paper](https://www.nature.com/articles/s41586-024-07421-0)][[code](https://github.com/jlko/semantic_uncertainty)]
  - Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. _Kuhn, Lorenz, Yarin Gal, and Sebastian Farquhar._ arXiv preprint arXiv:2302.09664 (2023) [[paper](https://arxiv.org/pdf/2302.09664)][[code](https://github.com/lorenzkuhn/semantic_uncertainty)]

- **Semantic Density.** Semantic Density: Uncertainty Quantification in Semantic Space for Large Language Models. _Qiu, Xin, and Risto Miikkulainen_ arXiv e-prints (2024): arXiv-2405.[[paper](https://arxiv.org/pdf/2405.13845)][[code](https://github.com/cognizant-ai-labs/semantic-density-paper)]

- **Uncertainty-aware Beam Search.** On hallucination and predictive uncertainty in conditional language generation. _Xiao, Yijun, and William Yang Wang_ arXiv preprint arXiv:2103.15025 (2021).[[paper](https://arxiv.org/pdf/2103.15025)][[code](https://github.com/iinemo/lm-polygraph?tab=readme-ov-file)]

- **Predictive variance.** Quantifying uncertainties in natural language processing tasks. _Xiao, Yijun, and William Yang Wang_  Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019. [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/4719/4597)]

- **Uncertainty-aware self-correction.** Improving the reliability of large language models by leveraging uncertainty-aware in-context learning. _Yang, Yuchen, et al._ arXiv preprint arXiv:2310.04782 (2023).[[paper](https://arxiv.org/pdf/2310.04782)]

- **Uncertainty Decomposition.** Uncertainty quantification for in-context learning of large language models. _Ling, Chen, et al._ arXiv preprint arXiv:2402.10189 (2024). [[paper](https://arxiv.org/pdf/2402.10189)][[code](https://github.com/lingchen0331/UQ_ICL)]

- **Uncertainty-aware Instruction Tuning.** Can LLMs learn uncertainty on their own? expressing uncertainty effectively in a self-training manner. _Liu, Shudong, et al_ Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.[[paper](https://aclanthology.org/2024.emnlp-main.1205.pdf)][[code](https://github.com/NLP2CT/UaIT)]

- **Uncertinty indices.** Quantifying uncertainty: a new era of measurement through large language models. _Audrino, Francesco, Jessica Gentner, and Simon Stalder_ (2024).[[paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4998319)]

- **Shifting Attention to more Relevant.** Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large language models. _Duan, Jinhao, et al._ arXiv preprint arXiv:2307.01379 (2023). [[paper](https://arxiv.org/pdf/2307.01379)][[code](https://github.com/jinhaoduan/SAR)]

- **Claim-conditioned probability.** Fact-checking the output of large language models via token-level uncertainty quantification. _Fadeeva, Ekaterina, et al._ arXiv preprint arXiv:2403.04696 (2024). [[paper](https://arxiv.org/pdf/2403.04696)][[code](https://github.com/IINemo/lm-polygraph)]

- **Hybrid uncertainty quantification.** Hybrid uncertainty quantification for selective text classification in ambiguous tasks. _Vazhentsev, Artem, et al._ Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023. [[paper](https://aclanthology.org/2023.acl-long.652.pdf)][[code](https://github.com/AIRI-Institute/hybrid_uncertainty_estimation)]

#### Conformal prediction

- **KnowNo.** Robots that ask for help: Uncertainty alignment for large language model planners. _Ren, Allen Z., et al._ arXiv preprint arXiv:2307.01928 (2023).[[paper](https://arxiv.org/pdf/2307.01928)][[code](https://robot-help.github.io/)]

- **Stopping rule.** Conformal language modeling. _Quach, Victor, et al._ arXiv preprint arXiv:2306.10193 (2023). [[paper](https://arxiv.org/pdf/2306.10193)][[code](https://github.com/Varal7/conformal-language-modeling)]

- **Exchangeability.** Conformal prediction with large language models for multi-choice question answering. _Kumar, Bhawesh, et al._ arXiv preprint arXiv:2305.18404 (2023). [[paper](https://arxiv.org/pdf/2305.18404)]

#### Language agents

- **CRITIC.** Critic: Large language models can self-correct with tool-interactive critiquing. _Gou, Zhibin, et al._ The Twelfth International Conference on Learning Representations (2024). [[paper](https://openreview.net/pdf?id=Sx038qxjek)][[code](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)]

- **Uncertainty-Aware Language Agent.** Towards uncertainty-aware language agent. _Han, Jiuzhou, Wray Buntine, and Ehsan Shareghi_ arXiv preprint arXiv:2401.14016 (2024). [[paper](https://arxiv.org/pdf/2401.14016)][[code](https://uala-agent.github.io/)]

- **Multi-agent Debate.** Encouraging divergent thinking in large language models through multi-agent debate. _Liang, Tian, et al._ arXiv preprint arXiv:2305.19118 (2023).[[paper](https://arxiv.org/pdf/2305.19118)][[code](https://github.com/Skytliang/Multi-Agents-Debate)]

- **Reflexion.** Reflexion: Language agents with verbal reinforcement learning. _Shinn, Noah, et al._ Advances in Neural Information Processing Systems 36 (2023): 8634-8652. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf)][[code](https://github.com/noahshinn024/reflexion)]

- **DebUnc.** DebUnc: mitigating hallucinations in large language model agent communication with uncertainty estimations. _Yoffe, Luke, Alfonso Amayuelas, and William Yang Wang_ arXiv preprint arXiv:2407.06426 (2024). [[paper](https://arxiv.org/pdf/2407.06426)][[code](https://github.com/lukeyoffe/debunc)]

- **ReConcile.** Reconcile: Round-table conference improves reasoning via consensus among diverse llms. _Chen, Justin Chih-Yao, Swarnadeep Saha, and Mohit Bansal_ arXiv preprint arXiv:2309.13007 (2023). [[paper](https://arxiv.org/pdf/2309.13007)] [[code](https://github.com/dinobby/ReConcile)]

#### Human uncertainty

- **Fluency and adecuacy.** Prevent the language model from being overconfident in neural machine translation. _Miao, Mengqi, et al._ arXiv preprint arXiv:2105.11098 (2021). [[paper](https://arxiv.org/pdf/2105.11098)][[code](https://github.com/Mlair77/nmt_adequacy)]

- **Human perceptual similary.** Human uncertainty makes classification more robust. _Peterson, Joshua C., et al._ Proceedings of the IEEE/CVF international conference on computer vision. 2019.[[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Peterson_Human_Uncertainty_Makes_Classification_More_Robust_ICCV_2019_paper.pdf)]

- **Human Distribution Calibration Error.** Stop measuring calibration when humans disagree. _Baan, Joris, et al._ arXiv preprint arXiv:2210.16133 (2022). [[paper](https://arxiv.org/pdf/2210.16133)]

- **Local intrinsic dimensions.** Characterizing truthfulness in large language model generations with local intrinsic dimension. _Yin, Fan, Jayanth Srinivasa, and Kai-Wei Chang_ arXiv preprint arXiv:2402.18048 (2024). [[paper](https://arxiv.org/pdf/2402.18048)][[code](https://github.com/fanyin3639/LID-HallucinationDetection)]

- **Active-prompt.** Active prompting with chain-of-thought for large language models. _Diao, Shizhe, et al._ arXiv preprint arXiv:2302.12246 (2023).[[paper](https://arxiv.org/pdf/2302.12246)][[code](https://github.com/shizhediao/active-prompt)]

#### Sampling-based (Closed-box)

- **Prompt ensemble.** Prompting gpt-3 to be reliable. _Si, Chenglei, et al._ arXiv preprint arXiv:2210.09150 (2022). [[paper](https://arxiv.org/pdf/2210.09150)][[code](https://github.com/NoviScl/GPT3-Reliability)]

- **Uncertainty Tripartite Testing Paradigm.** Unlocking the Power of LLM Uncertainty for Active In-Context Example Selection. _Huang, Hsiu-Yuan, et al._ arXiv preprint arXiv:2408.09172 (2024). [[paper](https://arxiv.org/pdf/2408.09172?)]

- **LUQ-ensemble.** Luq: Long-text uncertainty quantification for llms. _Zhang, Caiqi, et al._ arXiv preprint arXiv:2403.20279 (2024).[[paper](https://arxiv.org/pdf/2403.20279)][[code](https://github.com/caiqizh/LUQ)]

- **Probing/CoT uncertainty.** Quantifying uncertainty in natural language explanations of large language models. _Tanneru, Sree Harsha, Chirag Agarwal, and Himabindu Lakkaraju_ International Conference on Artificial Intelligence and Statistics. PMLR, 2024. [[paper](https://proceedings.mlr.press/v238/harsha-tanneru24a/harsha-tanneru24a.pdf)][[code](https://github.com/harsha070/uncertainty-quantification-nle)]

#### Semantic uncertainty

- **Kernel language entropy.** Kernel language entropy: Fine-grained uncertainty quantification for llms from semantic similarities. _Nikitin, Alexander, et al._ Advances in Neural Information Processing Systems 37 (2024): 8901-8929. [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/10c456d2160517581a234dfde15a7505-Paper-Conference.pdf)][[code]()]
- **Eccentricity/Degree Matrix/EigV.** Generating with confidence: Uncertainty quantification for black-box large language models. _Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun_ arXiv preprint arXiv:2305.19187 (2023).[[paper](https://arxiv.org/pdf/2305.19187)][[code](https://github.com/zlin7/UQ-NLG)]
- **Semantic embedding.** Improving Uncertainty Quantification in Large Language Models via Semantic Embeddings. _Grewal, Yashvir S., Edwin V. Bonilla, and Thang D. Bui_ arXiv preprint arXiv:2410.22685 (2024).[[paper](https://arxiv.org/pdf/2410.22685)]
- **Rowen.** Retrieve only when it needs: Adaptive retrieval augmentation for hallucination mitigation in large language models. _Ding, Hanxing, et al._ arXiv preprint arXiv:2402.10612 (2024).[[paper](https://arxiv.org/pdf/2402.10612)][[code](https://github.com/dhx20150812/Rowen)]

#### Self-Evaluation
  - **Self-refine.** Self-refine: Iterative refinement with self-feedback. _Madaan, Aman, et al._ Advances in Neural Information Processing Systems 36 (2024). [[paper](https://arxiv.org/pdf/2303.17651)][[code](https://github.com/madaan/self-refine)][[code](https://github.com/AlexanderVNikitin/kernel-language-entropy)]
  - **Self-verification.** Large language models are better reasoners with self-verification. _Weng, Yixuan, et al._ arXiv preprint arXiv:2212.09561 (2022).[[paper](https://arxiv.org/pdf/2212.09561)][[code](https://github.com/WENGSYX/Self-Verification)]
  - **SelfCheck.** Selfcheck: Using llms to zero-shot check their own step-by-step reasoning. _Miao, Ning, Yee Whye Teh, and Tom Rainforth_ arXiv preprint arXiv:2308.00436 (2023). [[paper](https://arxiv.org/pdf/2308.00436)][[code](https://github.com/NingMiao/SelfCheck)]
  - **SelfCheckGPT.** Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. _Manakul, Potsawee, Adian Liusie, and Mark JF Gales_ arXiv preprint arXiv:2303.08896 (2023). [[paper](https://arxiv.org/pdf/2303.08896)][[code](https://github.com/potsawee/selfcheckgpt)]
  - **Deductive Reasoning.** Deductive verification of chain-of-thought reasoning. _Ling, Zhan, et al._ Advances in Neural Information Processing Systems 36 (2023): 36407-36433. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/72393bd47a35f5b3bee4c609e7bba733-Paper-Conference.pdf)][[code](https://github.com/lz1oceani/verify_cot)]
  - **Reference Overlap.** Do Language Models Know When They're Hallucinating References? _Agrawal, Ayush, et al._  arXiv preprint arXiv:2305.18248 (2023). [[paper](https://arxiv.org/pdf/2305.18248)][[code](https://github.com/microsoft/hallucinated-references)]

#### Self-detection
  - **Answer clustering.** Knowing what llms do not know: A simple yet effective self-detection method. _Zhao, Yukun, et al._ arXiv preprint arXiv:2310.17918 (2023). [[paper](https://arxiv.org/pdf/2310.17918)][[code](https://github.com/yukunZhao/Self-DETECTION)]
  - **BSDetector.** Quantifying uncertainty in answers from any language model and enhancing their trustworthiness. _Chen, Jiuhai, and Jonas Mueller_ arXiv preprint arXiv:2308.16175 (2023). [[paper](https://arxiv.org/pdf/2308.16175)]
  - **Joint confidence.** Think twice before trusting: Self-detection for large language models through comprehensive answer reflection. _Li, Moxin, et al._ arXiv preprint arXiv:2403.09972 (2024). [[paper](https://arxiv.org/pdf/2403.09972)]
  - **Self-reflection.** Towards mitigating hallucination in large language models via self-reflection. _Ji, Ziwei, et al._ arXiv preprint arXiv:2310.06271 (2023).[[paper](https://arxiv.org/pdf/2310.06271)][[code](https://github.com/ziweiji/Self_Reflection_Medical)]

#### Verbalized uncertainty
  - **Epistemic markers.** Navigating the grey area: How expressions of uncertainty and overconfidence affect language models. _Zhou, Kaitlyn, Dan Jurafsky, and Tatsunori Hashimoto_ arXiv preprint arXiv:2302.13439 (2023) [[paper](https://aclanthology.org/2023.emnlp-main.335.pdf)][[code](https://github.com/katezhou/navigating_the_grey/tree/main)]
  - **Stable explanation.** Cycles of thought: Measuring llm confidence through stable explanations. _Becker, Evan, and Stefano Soatto_ arXiv preprint arXiv:2406.03441 (2024).[[paper](https://arxiv.org/pdf/2406.03441)]
  - **Demonstration uncertainty.** Improving open information extraction with large language models: A study on demonstration uncertainty. _Ling, Chen, et al._ arXiv preprint arXiv:2309.03433 (2023).[[paper](https://arxiv.org/pdf/2309.03433)][[code](https://github.com/lingchen0331/demonstration_uncertainty)]
  - **Red teaming.** Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. _Ganguli, Deep, et al._ arXiv preprint arXiv:2209.07858 (2022). [[paper](https://arxiv.org/pdf/2209.07858)][[code](https://github.com/anthropics/hh-rlhf)]
  - **Convex hull area.** Uncertainty quantification in large language models through convex hull analysis. _Catak, Ferhat Ozgur, and Murat Kuzlu_ Discover Artificial Intelligence 4.1 (2024): 1-14. [[paper](https://arxiv.org/pdf/2406.19712v1)]

### Calibration

#### Bias reduction

- **In-context learning**
  - Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering. _Zhou, Han, et al._ The Twelfth International Conference on Learning Representations. 2024. [[paper](https://arxiv.org/pdf/2309.17249)][[code](https://github.com/cambridgeltl/ClaPS/blob/main/algs/test_time_bn.py)]
  - Prototypical Calibration for Few-shot Learning of Language Models _Zhixiong Han and Yaru Hao and Li Dong and Yutao Sun and Furu Wei_ The Eleventh International Conference on Learning Representations. 2023.[[paper](https://openreview.net/forum?id=nUsP9lFADUF)][[code](https://github.com/ZihanWangKi/x-TC/blob/main/external/prompt_gpt/ProtoCal.py)]
  - Mitigating label biases for in-context learning. _Fei, Yu, et al._ Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023. [[paper](https://aclanthology.org/2023.acl-long.783.pdf)][[code](https://github.com/fywalter/label-bias)]
  - Answer-level calibration for free-form multiple choice question answering. _Kumar, Sawan._ Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022.[[paper](https://aclanthology.org/2022.acl-long.49.pdf)][[code](https://github.com/SawanKumar28/alc)]
  - Calibrate before use: Improving few-shot performance of language models. _Zhao, Zihao, et al._ International conference on machine learning. PMLR, 2021.[[paper](https://arxiv.org/pdf/2102.09690)][[code](https://github.com/tonyzhaozh/few-shot-learning/tree/main?tab=readme-ov-file)]
  - Surface form competition: Why the highest probability answer isn't always right. _Holtzman, Ari, et al._ arXiv preprint arXiv:2104.08315 (2021).[[paper](https://arxiv.org/pdf/2104.08315)][[code](https://github.com/peterwestuw/surface-form-competition)]

- **Prompt optimization.** Survival of the most influential prompts: Efficient black-box prompt search via clustering and pruning. _Zhou, Han, et al._ arXiv preprint arXiv:2310.12774 (2023). [[paper](https://arxiv.org/pdf/2310.12774)][[code](https://github.com/cambridgeltl/ClaPS)]

- **Multi-calibration**
  - Multicalibration for confidence scoring in llms. _Detommaso, Gianluca, et al._ arXiv preprint arXiv:2404.04689 (2024). [[paper](https://arxiv.org/abs/2404.04689)]
  - Multicalibration: Calibration for the (computationally-identifiable) masses. _Hébert-Johnson, Ursula, et al._ International Conference on Machine Learning. PMLR, 2018. [[paper](http://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf)]
  - Low-degree multicalibration. _Gopalan, Parikshit, et al._ Conference on Learning Theory. PMLR, 2022.[[paper](https://proceedings.mlr.press/v178/gopalan22a/gopalan22a.pdf)]
  - Loss minimization yields multicalibration for large neural networks. _Błasiok, Jarosław, et al._ 15th Innovations in Theoretical Computer Science Conference (ITCS 2024). Vol. 287. Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2024.[[paper](https://arxiv.org/abs/2304.09424)]
  - Calibrating predictions to decisions: A novel approach to multi-class calibration. _Zhao, Shengjia, et al._ Advances in Neural Information Processing Systems 34 (2021): 22313-22324.[[paper](https://proceedings.neurips.cc/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf)]

- **Selection bias.** Large language models are not robust multiple choice selectors. _Zheng, Chujie, et al._ arXiv preprint arXiv:2309.03882 (2023).[[paper](https://arxiv.org/pdf/2309.03882)][[code](https://github.com/chujiezheng/LLM-MCQ-Bias)]

#### Open-Box calibration

- **Label smoothing**
  - On the inference calibration of neural machine translation. _Wang, Shuo, et al._ arXiv preprint arXiv:2005.00963 (2020).[[paper](https://arxiv.org/pdf/2005.00963)][[code](https://github.com/shuo-git/InfECE)]
  - Calibration of pre-trained transformers. _Desai, Shrey, and Greg Durrett_ arXiv preprint arXiv:2003.07892 (2020). [[paper](https://arxiv.org/pdf/2003.07892)][[code](https://github.com/shreydesai/calibration)]
  - Learning confidence for transformer-based neural machine translation. _Lu, Yu, et al._ arXiv preprint arXiv:2203.11413 (2022).[[paper](https://arxiv.org/pdf/2203.11413)][[code](https://github.com/yulu-dada/Learned-conf-NMT)]
  - Adaptive label smoothing with self-knowledge in natural language generation. _Lee, Dongkyu, Ka Chun Cheung, and Nevin L. Zhang_ arXiv preprint arXiv:2210.13459 (2022).[[paper](https://arxiv.org/pdf/2210.13459)][[code]()]
  - When does label smoothing help? _Müller, Rafael, Simon Kornblith, and Geoffrey E. Hinton._ Advances in neural information processing systems 32 (2019). [[paper](https://arxiv.org/pdf/1906.02629v3)][[code](https://github.com/seominseok0429/label-smoothing-visualization-pytorch)]

- **Sequence likelihood calibration**
  - Calibrating sequence likelihood improves conditional language generation. _Zhao, Yao, et al._  The Eleventh International Conference on Learning Representations. 2023. [[paper](https://openreview.net/pdf?id=0qSOodKmJaN)]
  - SLiC-HF: Sequence likelihood calibration with human feedback. _Zhao, Yao, et al._ arXiv preprint arXiv:2305.10425 (2023).[[paper](https://arxiv.org/pdf/2305.10425)][[huggingface](https://huggingface.co/papers/2305.10425)]
  - How can we know when language models know? on the calibration of language models for question answering. _Jiang, Zhengbao, et al._ Transactions of the Association for Computational Linguistics 9 (2021): 962-977.[[paper](https://aclanthology.org/2021.tacl-1.57.pdf)][[code](https://github.com/jzbjyb/lm-calibration)]

- **Sayself.** Sayself: Teaching llms to express confidence with self-reflective rationales. _Xu, Tianyang, et al._ Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.[[paper](https://aclanthology.org/2024.emnlp-main.343.pdf)][[code](https://github.com/xu1868/SaySelf)]

- **Decision-based RL.** Linguistic calibration of long-form generations. _Band, Neil, et al._ arXiv preprint arXiv:2404.00474 (2024). [[paper](https://arxiv.org/pdf/2404.00474)][[code](https://github.com/tatsu-lab/linguistic_calibration)]

- **Refusal-Aware Instruction Tuning.** R-tuning: Instructing large language models to say ‘I don’t know’. _Zhang, Hanning, et al._ Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024. [[paper](https://aclanthology.org/2024.naacl-long.394.pdf)][[code](https://github.com/shizhediao/R-Tuning)]

- **Listener-Aware Calibration for Implicit and Explicit confidence.** LACIE: Listener-aware finetuning for calibration in large language models. _Stengel-Eskin, Elias, Peter Hase, and Mohit Bansal_ Advances in Neural Information Processing Systems 37 (2024): 43080-43106.[[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/4b8eaf3bcdc105423a972ed90eb07217-Paper-Conference.pdf)][[code](https://github.com/esteng/pragmatic_calibration)]

- **Calibration tuning.** Calibration-tuning: Teaching large language models to know what they don’t know. _Kapoor, Sanyam, et al._ Proceedings of the 1st Workshop on Uncertainty-Aware NLP (UncertaiNLP 2024). 2024.[[paper](https://aclanthology.org/2024.uncertainlp-1.1.pdf)]

- **Low-Rank Adaptation (LoRA) ensemble.** Uncertainty quantification in fine-tuned LLMs using LoRA ensembles. _Balabanov, Oleksandr, and Hampus Linander_ arXiv preprint arXiv:2402.12264 (2024). [[paper](https://arxiv.org/pdf/2402.12264)][[code](https://github.com/oleksandr-balabanov/equivariant-posteriors/tree/master/experiments/lora_ensembles)]

- **Laplace-LoRA.** Bayesian low-rank adaptation for large language models. _Yang, Adam X., et al._ arXiv preprint arXiv:2308.13111 (2023). [[paper](https://arxiv.org/pdf/2308.13111)][[code](https://github.com/MaximeRobeyns/bayesian_lora)]

- **LitCab.** Litcab: Lightweight language model calibration over short-and long-form responses. _Liu, Xin, Muhammad Khalifa, and Lu Wang_ arXiv preprint arXiv:2310.19208 (2023). [[paper](https://arxiv.org/pdf/2310.19208)][[code](https://github.com/launchnlp/LitCab)]

- **ActCab.** Enhancing language model factuality via activation-based confidence calibration and guided decoding. _Liu, Xin, Farima Fatahi Bayat, and Lu Wang_ arXiv preprint arXiv:2406.13230 (2024).[[paper](https://arxiv.org/pdf/2406.13230)][[code](https://github.com/launchnlp/ActCab)]

- **Early existing.** Confident adaptive language modeling. _Schuster, Tal, et al._ Advances in Neural Information Processing Systems 35 (2022): 17456-17472.[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/6fac9e316a4ae75ea244ddcef1982c71-Paper-Conference.pdf)][[code](https://github.com/google-research/t5x/tree/main/t5x/contrib/calm)]

- **CaliNet.** Calibrating factual knowledge in pretrained language models. _Dong, Qingxiu, et al._ arXiv preprint arXiv:2210.03329 (2022).[[paper](https://arxiv.org/pdf/2210.03329)][[code](https://github.com/dqxiu/CaliNet)]

#### Post-hoc Calibration Methods

- **Temperature scaling.** On calibration of modern neural networks. _Guo, Chuan, et al._ International conference on machine learning. PMLR 2017 [[paper](https://arxiv.org/pdf/1706.04599)][[code](https://github.com/gpleiss/temperature_scaling.git)]

- **Platt scaling.** Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. _Platt, John._ Advances in large margin classifiers 10.3 (1999): 61-74. [[paper](https://www.researchgate.net/profile/John-Platt-2/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods/links/004635154cff5262d6000000/Probabilistic-Outputs-for-Support-Vector-Machines-and-Comparisons-to-Regularized-Likelihood-Methods.pdf)]

- **Atypicality.** Beyond confidence: Reliable models should also consider atypicality. _Yuksekgonul, Mert, et al._ Advances in Neural Information Processing Systems 36 (2023): 38420-38453.[[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7900318ffaf5e9bc60250f134c6cc3c7-Paper-Conference.pdf)][[code](https://github.com/mertyg/beyond-confidence-atypicality)]

- **Joint pipeline calibration.** Calibration of machine reading systems at scale. _Dhuliawala, Shehzaad, et al._ arXiv preprint arXiv:2203.10623 (2022).[[paper](https://arxiv.org/pdf/2203.10623)][[code]()]

- **PLEX.** Plex: Towards reliability using pretrained large model extensions. _Tran, Dustin, et al._ arXiv preprint arXiv:2207.07411 (2022). [[paper](https://arxiv.org/pdf/2207.07411)][[code](https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/plex.py)]

- **Histogram binning.** Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. _Zadrozny, Bianca, and Charles Elkan_ Icml. Vol. 1. No. 05. 2001.[[paper](http://cseweb.ucsd.edu/~elkan/calibrated.pdf)]

- **Isotonic regression.** Transforming classifier scores into accurate multiclass probability estimates. _Zadrozny, Bianca, and Charles Elkan_ Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. 2002.[[paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=04e10f745a7267453788a22f5150b5a32b2b3951)]

- **Rank calibration.** Uncertainty in language models: Assessment through rank-calibration. _Huang, Xinmeng, et al._ arXiv preprint arXiv:2404.03163 (2024). [[paper](https://arxiv.org/pdf/2404.03163)][[code](https://github.com/shuoli90/Rank-Calibration/tree/main)]

- **Bayesian Binning into Quantiles (BBQ).** Obtaining well calibrated probabilities using bayesian binning. _Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht._ Proceedings of the AAAI conference on artificial intelligence. Vol. 29. No. 1. 2015. [[paper](https://dl.acm.org/doi/10.5555/2888116.2888120)][[code](https://github.com/pakdaman/calibration)]

- **Scale-binning calibrator.** Verified uncertainty calibration. _Kumar, Ananya, Percy S. Liang, and Tengyu Ma._ Advances in Neural Information Processing Systems 32 (2019).[[paper](https://arxiv.org/pdf/1909.10155)][[code](https://github.com/p-lambda/verified_calibration)]

#### Closed-Box Calibration

- **Linguistic calibration**
  - Navigating the grey area: How expressions of uncertainty and overconfidence affect language models. _Zhou, Kaitlyn, Dan Jurafsky, and Tatsunori Hashimoto_ arXiv preprint arXiv:2302.13439 (2023) [[paper](https://aclanthology.org/2023.emnlp-main.335.pdf)][[code](https://github.com/katezhou/navigating_the_grey/tree/main)]
  - Fact-and-reflection (far) improves confidence calibration of large language models. _Zhao, Xinran, et al._ arXiv preprint arXiv:2402.17124 (2024). [[paper](https://arxiv.org/pdf/2402.17124)][[code](https://github.com/colinzhaoust/fact-and-reflection)]
  - Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty. _Zhou, Kaitlyn, et al._ arXiv preprint arXiv:2401.06730 (2024). [[paper](https://arxiv.org/pdf/2401.06730)]
  - Calibrating the confidence of large language models by eliciting fidelity. _Zhang, Mozhi, et al._ arXiv preprint arXiv:2404.02655 (2024). [[paper](https://arxiv.org/pdf/2404.02655?)]
  - Reducing conversational agents’ overconfidence through linguistic calibration. _Mielke, Sabrina J., et al._ Transactions of the Association for Computational Linguistics 10 (2022): 857-872. [[paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00494/112606/Reducing-Conversational-Agents-Overconfidence)][[data](https://parl.ai/projects/metacognition/)]
  - Teaching models to express their uncertainty in words. _Lin, Stephanie, Jacob Hilton, and Owain Evans._ arXiv preprint arXiv:2205.14334 (2022).[[paper](https://arxiv.org/pdf/2205.14334)][[code](https://github.com/sylinrl/calibratedmath)]
  - Atomic Calibration of LLMs in Long-Form Generations. _Zhang, Caiqi, et al._ arXiv preprint arXiv:2410.13246 (2024).[[paper](https://arxiv.org/pdf/2410.13246)][[code]()]

- **Human confidence.**
  - What large language models know and what people think they know. _Steyvers, Mark, et al._ Nature Machine Intelligence (2025): 1-11.[[paper](https://www.nature.com/articles/s42256-024-00976-7.pdf)]
  - The calibration gap between model and human confidence in large language models. _Steyvers, Mark, et al._ arXiv preprint arXiv:2401.13835 (2024). [[paper](https://arxiv.org/pdf/2401.13835)][[code](https://osf.io/y7pr6/)]

- **Self-consistency**
  - Self-consistency improves chain of thought reasoning in language models. _Wang, Xuezhi, et al._ The Eleventh International Conference on Learning Representations (2023). [[paper](https://openreview.net/pdf?id=1PL1NIMMrw)][[code](https://github.com/codelion/optillm/blob/main/optillm/self_consistency.py)]
  - Calibrating long-form generations from large language models. _Huang, Yukun, et al._ arXiv preprint arXiv:2402.06544 (2024). [[paper](https://arxiv.org/pdf/2402.06544)][[code](https://github.com/kkkevinkkkkk/calibration)]
  - Confident adaptive language modeling. _Schuster, Tal, et al._ Advances in Neural Information Processing Systems 35 (2022): 17456-17472.[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/6fac9e316a4ae75ea244ddcef1982c71-Paper-Conference.pdf)][[code](https://github.com/google-research/t5x/tree/main/t5x/contrib/calm)]
  - Universal self-consistency for large language model generation. _Chen, Xinyun, et al._ arXiv preprint arXiv:2311.17311 (2023). [[paper](https://arxiv.org/pdf/2311.17311)]
  - Exploring large language models for multi-modal out-of-distribution detection. _Dai, Yi, et al._ arXiv preprint arXiv:2310.08027 (2023).[[paper](https://arxiv.org/pdf/2310.08027)]
  - Combining confidence elicitation and sample-based methods for uncertainty quantification in misinformation mitigation. _Rivera, Mauricio, et al._ arXiv preprint arXiv:2401.08694 (2024).[[paper](https://arxiv.org/pdf/2401.08694)]
  - Verify when Uncertain: Beyond Self-Consistency in Black Box Hallucination Detection. _Xue, Yihao, et al._ arXiv preprint arXiv:2502.15845 (2025). [[paper](https://arxiv.org/pdf/2502.15845?)]

- **Ensemble**
  - Calibrating language models via augmented prompt ensembles. _Jiang, Mingjian, et al._ (2023). [[paper](https://openreview.net/pdf?id=L0dc4wqbNs)]
  - On task performance and model calibration with supervised and self-ensembled in-context learning. _Li, Chengzu, et al._ arXiv preprint arXiv:2312.13772 (2023).[[paper](https://arxiv.org/pdf/2312.13772)][[code](https://github.com/cambridgeltl/ensembled-sicl)]

- **Auxiliary model**
  - Calibrating Large Language Models Using Their Generations Only _Ulmer, Dennis, et al._ arXiv preprint arXiv:2403.05973 (2024). [[paper](https://arxiv.org/pdf/2403.05973)][[code](https://github.com/parameterlab/apricot/blob/main/README.md?plain=1)]
  - Llamas Know What GPTs Don't Show: Surrogate Models for Confidence Estimation. _Shrivastava, Vaishnavi, Percy Liang, and Ananya Kumar_  arXiv preprint arXiv:2311.08877 (2023).[[paper](https://arxiv.org/pdf/2311.08877)]
  - Lm vs lm: Detecting factual errors via cross examination. _Cohen, Roi, et al._ arXiv preprint arXiv:2305.13281 (2023).[[paper](https://arxiv.org/pdf/2305.13281)]
  - The internal state of an LLM knows when it's lying. _Azaria, Amos, and Tom Mitchell_ arXiv preprint arXiv:2304.13734 (2023).[[paper](https://arxiv.org/pdf/2304.13734)][[code](azariaa.com/Content/Datasets/true-false-dataset.zip)]

### Metrics
- **Recall@1 AUROC.** Url: A representation learning benchmark for transferable uncertainty estimates. _Kirchhof, Michael, et al._ Advances in Neural Information Processing Systems 36 (2023): 13956-13980.[[paper](https://arxiv.org/pdf/2307.03810)][[code](https://github.com/mkirchhof/url/tree/url_at_time_of_submission)]

- **Expected Calibration Error (ECE)**
  - Well-calibrated model uncertainty with temperature scaling for dropout variational inference. _Laves, Max-Heinrich, et al._ arXiv preprint arXiv:1909.13550 (2019). [[paper](https://arxiv.org/pdf/1909.13550)][[code](https://github.com/mlaves/bayesian-temperature-scaling)]
  - Obtaining well calibrated probabilities using bayesian binning. _Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht._ Proceedings of the AAAI conference on artificial intelligence. Vol. 29. No. 1. 2015. [[paper](https://dl.acm.org/doi/10.5555/2888116.2888120)][[code](https://github.com/pakdaman/calibration/blob/master/BBQ/getECE.m)]

- **Uncertainty Calibration Error (UCE).** Well-calibrated model uncertainty with temperature scaling for dropout variational inference. _Laves, Max-Heinrich, et al._ arXiv preprint arXiv:1909.13550 (2019). [[paper](https://arxiv.org/pdf/1909.13550)][[code](https://github.com/mlaves/bayesian-temperature-scaling)]

- **Class-wise ECE.** Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration.  Advances in neural information processing systems 32 (2019). _Kull, Meelis, et al._ [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf)][[code](https://github.com/dirichletcal/experiments_neurips/blob/master/calib/models/dirichlet_keras.py)]

- **Local Calibration Error (LCE).** Local calibration: metrics and recalibration. _Luo, Rachel, et al._ Uncertainty in Artificial Intelligence. PMLR, 2022.[[paper](https://proceedings.mlr.press/v180/luo22a/luo22a.pdf)]

- **Smooth ECE.** Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing. The Twelfth International Conference on Learning Representations. _Blasiok, Jaroslaw, and Preetum Nakkiran_ [[paper](https://arxiv.org/pdf/2309.12236)][[code](https://github.com/apple/ml-calibration )]

- **Accuracy vs Uncertainty Calibration(AvUC).** Improving model calibration with accuracy versus uncertainty optimization. _Krishnan, Ranganath, and Omesh Tickoo._ Advances in Neural Information Processing Systems 33 (2020): 18237-18248. [[paper](https://papers.nips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf)][[code](https://github.com/IntelLabs/AVUC)]

- **Patch Accuracy vs Patch Uncertainty (PAvPU).** Evaluating bayesian deep learning methods for semantic segmentation. _Mukhoti, Jishnu, and Yarin Gal._ arXiv preprint arXiv:1811.12709 (2018).[[paper](https://arxiv.org/pdf/1811.12709)][[code](https://github.com/IntelLabs/AVUC)]

- **Trust Score.** To trust or not to trust a classifier. _Jiang, Heinrich, et al._ Advances in neural information processing systems 31 (2018).[[paper](https://arxiv.org/pdf/1805.11783)][[code](https://github.com/google/TrustScore)]

- **Sequence BLEU.** Analyzing uncertainty in neural machine translation. _Ott, Myle, et al._ International Conference on Machine Learning. PMLR, 2018. [[paper](https://arxiv.org/pdf/1803.00047)][[code](https://github.com/facebookresearch/analyzing-uncertainty-nmt)]

### Evaluation
  - **Top-k verbalized uncertainty.** Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. _Tian, Katherine, et al._  arXiv preprint arXiv:2305.14975 (2023).[[paper](https://arxiv.org/pdf/2305.14975)]
  - **Top-k + Self-consistency.** Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. _Xiong, Miao, et al._  arXiv preprint arXiv:2306.13063 (2023) [[paper](https://arxiv.org/pdf/2306.13063)][[code](https://github.com/MiaoXiong2320/llm-uncertainty)]

### Self-correction
- **Reasoning.** Large language models cannot self-correct reasoning yet. _Huang, Jie, et al._ The Twelfth International Conference on Learning Representations (2024).[[paper](https://openreview.net/pdf?id=IkmD3fKBPQ)]

### Classical Deep Learning Methods

- **Bayesian Neural Networks (BNN)**
  - `methods/bayesian/bayesian_conv.py`
  - Bayesian active learning for classification and preference learning. _Houlsby, Neil, et al._ arXiv preprint arXiv:1112.5745 (2011). [[paper](https://arxiv.org/pdf/1112.5745)][[code](https://github.com/AIRI-Institute/al_toolbox/blob/main/acleto/al4nlp/query_strategies/bald.py)]

- **Negative Log Likelihood (NLL)**
  - `methods/nentr/NENTR.py`

- **Similarity Sensitive Entropy.** Measuring Uncertainty in Neural Machine Translation with Similarity-Sensitive Entropy. _Cheng, Julius, and Andreas Vlachos._ Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.[[paper](https://aclanthology.org/2024.eacl-long.129.pdf)][[code](https://github.com/juliusc/s3e)]

- **Deep Ensembles.** Simple and scalable predictive uncertainty estimation using deep ensembles. _Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell._ Advances in neural information processing systems 30 (2017). [[paper](https://arxiv.org/pdf/1612.01474)][[code](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation)]

- **TransCal.** Transferable calibration with lower bias and variance in domain adaptation. _Wang, Ximei, et al._ Advances in Neural Information Processing Systems 33 (2020): 19212-19223. [[paper](https://papers.nips.cc/paper/2020/hash/df12ecd077efc8c23881028604dbb8cc-Abstract.html)][[code](https://github.com/thuml/TransCal)]

- **Focal loss.** Calibrating deep neural networks using focal loss. _Mukhoti, Jishnu, et al._ Advances in Neural Information Processing Systems 33 (2020): 15288-15299.[[paper](https://arxiv.org/pdf/2002.09437)][[code](https://github.com/torrvision/focal_calibration)]

## Contributing

For contributions, please open an issue or submit a pull request.
