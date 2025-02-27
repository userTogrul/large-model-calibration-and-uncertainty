# Measuring Reliability of Large Language Models

\
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![last commit](https://img.shields.io/github/last-commit/userTogrul/large-model-calibration-and-uncertainty?color=green)](https://img.shields.io/github/last-commit/userTogrul/large-model-calibration-and-uncertainty?color=green)
[![Paper: arXiv](https://img.shields.io/badge/Paper-arXiv-%23B31B1B)](https://github.com/userTogrul/large-model-calibration-and-uncertainty/tree/main)
[![Made with Love](https://img.shields.io/badge/Made%20with-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)

This reposity contains papers and source codes for various methods of quantification of uncertainty and calibration in large models that can be important to consider for research community.

## Setup

Please download and install conda for python environment, and setup necessessary libraries:

    conda create -n large-model-calibration-and-uncertainty python=3.10
    python -m pip install -r requirements.txt

Then create a `secret.py` file with necessary api keys as described in `secret_template.py` in the same folder as `main.py`.

## Usage

To run a script for **Llama-3.1-8B-Instruct**, use the following command:

    sh run_llama.sh

type 'Yes' or 'y' when prompted.

For **Mistral-7B-Instruct-v0.3**, use the following command:

    sh run_mistral.sh

For **Qwen2.5-7B-Instruct**, use the following command:

    sh run_qwen.sh

For **gpt-4 & gpt-4o-mini**, use the following command:

    sh run_openai.sh

## Methods Included

### Uncertanty Quantification

- **Bayesian Neural Networks (BNN)**
  - `methods/bayesian/bayesian_conv.py`
  - Bayesian active learning for classification and preference learning. _Houlsby, Neil, et al._ arXiv preprint arXiv:1112.5745 (2011). [[paper](https://arxiv.org/pdf/1112.5745)][[code](https://github.com/AIRI-Institute/al_toolbox/blob/main/acleto/al4nlp/query_strategies/bald.py)]

- **Monte Carlo Dropout (MCD)**
  - Dropout as a bayesian approximation: Representing model uncertainty in deep learning. _Gal, Yarin, and Zoubin Ghahramani._ International conference on machine learning. PMLR, 2016. [[paper](https://arxiv.org/pdf/1506.02142)][[code](https://github.com/yaringal/DropoutUncertaintyExps/blob/master/readme.md)]

- **Negative Log Likelihood (NLL)**
  - `methods/nentr/NENTR.py`

- **Deep Ensembles**
  - Simple and scalable predictive uncertainty estimation using deep ensembles. _Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell._ Advances in neural information processing systems 30 (2017). [[paper](https://arxiv.org/pdf/1612.01474)][[code](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation)]

- **Deep Ensembles with Low-Rank Adaptation**
  - Uncertainty quantification in fine-tuned LLMs using LoRA ensembles. _Balabanov, Oleksandr, and Hampus Linander_ arXiv preprint arXiv:2402.12264 (2024). [[paper](https://arxiv.org/pdf/2402.12264)][[code](https://github.com/oleksandr-balabanov/equivariant-posteriors/tree/master/experiments/lora_ensembles)]

- **BatchEnsemble in Pre-training**
  - Plex: Towards reliability using pretrained large model extensions. _Tran, Dustin, et al._ arXiv preprint arXiv:2207.07411 (2022). [[paper](https://arxiv.org/pdf/2207.07411)][[code](https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/plex.py)]
  - BatchEnsemble: an Alternative Approach to Efficient Ensemble and Lifelong Learning. _Wen, Yeming, Dustin Tran, and Jimmy Ba._ International Conference on Learning Representations. 2020. [[paper](https://arxiv.org/pdf/2002.06715)][[code](https://github.com/google/edward2)]


- **Recall@1 AUROC**
  -  Url: A representation learning benchmark for transferable uncertainty estimates. _Kirchhof, Michael, et al._ Advances in Neural Information Processing Systems 36 (2023): 13956-13980.[[paper](https://arxiv.org/pdf/2307.03810)][[code](https://github.com/mkirchhof/url/tree/url_at_time_of_submission)]

- **Entropy-based**
  - Detecting hallucinations in large language models using semantic entropy. _Farquhar, Sebastian, et al._ Nature 630.8017 (2024): 625-630 [[paper](https://www.nature.com/articles/s41586-024-07421-0)][[code](https://github.com/jlko/semantic_uncertainty)]
  - Measuring Uncertainty in Neural Machine Translation with Similarity-Sensitive Entropy. _Cheng, Julius, and Andreas Vlachos._ Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.[[paper](https://aclanthology.org/2024.eacl-long.129.pdf)][[code](https://github.com/juliusc/s3e)]
  - Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. _Kuhn, Lorenz, Yarin Gal, and Sebastian Farquhar._ arXiv preprint arXiv:2302.09664 (2023) [[paper](https://arxiv.org/pdf/2302.09664)][[code](https://github.com/lorenzkuhn/semantic_uncertainty)]

- **Self-Evaluation**
  - Critic: Large language models can self-correct with tool-interactive critiquing. _Gou, Zhibin, et al._ The Twelfth International Conference on Learning Representations (2024). [[paper](https://openreview.net/pdf?id=Sx038qxjek)][[code](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)]
  - Large language models cannot self-correct reasoning yet. _Huang, Jie, et al._ The Twelfth International Conference on Learning Representations (2024).[[paper](https://openreview.net/pdf?id=IkmD3fKBPQ)]
  - Self-refine: Iterative refinement with self-feedback. _Madaan, Aman, et al._ Advances in Neural Information Processing Systems 36 (2024). [[paper](https://arxiv.org/pdf/2303.17651)][[code](https://github.com/madaan/self-refine)]
  - Language models (mostly) know what they know.  _Kadavath, Saurav, et al._ arXiv preprint arXiv:2207.05221 (2022).[[paper](https://arxiv.org/pdf/2207.05221)]

### Calibration
#### Metrics

- **Expected Calibration Error (ECE)**
  - Well-calibrated model uncertainty with temperature scaling for dropout variational inference. _Laves, Max-Heinrich, et al._ arXiv preprint arXiv:1909.13550 (2019). [[paper](https://arxiv.org/pdf/1909.13550)][[code](https://github.com/mlaves/bayesian-temperature-scaling)]
  - Obtaining well calibrated probabilities using bayesian binning. _Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht._ Proceedings of the AAAI conference on artificial intelligence. Vol. 29. No. 1. 2015. [[paper](https://dl.acm.org/doi/10.5555/2888116.2888120)][[code](https://github.com/pakdaman/calibration/blob/master/BBQ/getECE.m)]

- **Uncertainty Calibration Error (UCE)**
  - Well-calibrated model uncertainty with temperature scaling for dropout variational inference. _Laves, Max-Heinrich, et al._ arXiv preprint arXiv:1909.13550 (2019). [[paper](https://arxiv.org/pdf/1909.13550)][[code](https://github.com/mlaves/bayesian-temperature-scaling)]

- **Class-wise ECE**
  - Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration.  Advances in neural information processing systems 32 (2019). _Kull, Meelis, et al._ [[paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf)][[code](https://github.com/dirichletcal/experiments_neurips/blob/master/calib/models/dirichlet_keras.py)]

- **Local Calibration Error (LCE)**
  - Local calibration: metrics and recalibration. _Luo, Rachel, et al._ Uncertainty in Artificial Intelligence. PMLR, 2022.[[paper](https://proceedings.mlr.press/v180/luo22a/luo22a.pdf)]

- **Smooth ECE**
  - "Smooth ECE: Principled Reliability Diagrams via Kernel Smoothing." The Twelfth International Conference on Learning Representations. _Blasiok, Jaroslaw, and Preetum Nakkiran_ [[paper](https://arxiv.org/pdf/2309.12236)][[code](https://github.com/apple/ml-calibration )]

- **Accuracy vs Uncertainty Calibration(AvUC)**
  - Improving model calibration with accuracy versus uncertainty optimization. _Krishnan, Ranganath, and Omesh Tickoo._ Advances in Neural Information Processing Systems 33 (2020): 18237-18248. [[paper](https://papers.nips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf)][[code](https://github.com/IntelLabs/AVUC)]

- **Patch Accuracy vs Patch Uncertainty (PAvPU)**
  - Evaluating bayesian deep learning methods for semantic segmentation. _Mukhoti, Jishnu, and Yarin Gal._ arXiv preprint arXiv:1811.12709 (2018).[[paper](https://arxiv.org/pdf/1811.12709)][[code](https://github.com/IntelLabs/AVUC)]

- **Trust Score**
  - To trust or not to trust a classifier. _Jiang, Heinrich, et al._ Advances in neural information processing systems 31 (2018).[[paper](https://arxiv.org/pdf/1805.11783)][[code](https://github.com/google/TrustScore)]

#### Open-Box Calibration
- **Label Smoothing**
  - On the inference calibration of neural machine translation. _Wang, Shuo, et al._ arXiv preprint arXiv:2005.00963 (2020).[[paper](https://arxiv.org/pdf/2005.00963)][[code](https://github.com/shuo-git/InfECE)]
  - Calibration of pre-trained transformers. _Desai, Shrey, and Greg Durrett_ arXiv preprint arXiv:2003.07892 (2020). [[paper](https://arxiv.org/pdf/2003.07892)][[code](https://github.com/shreydesai/calibration)]
  - When does label smoothing help? _Müller, Rafael, Simon Kornblith, and Geoffrey E. Hinton._ Advances in neural information processing systems 32 (2019). [[paper](https://arxiv.org/pdf/1906.02629v3)][[code](https://github.com/seominseok0429/label-smoothing-visualization-pytorch)]

- **Fine-tuning**
  - Calibrating sequence likelihood improves conditional language generation. _Zhao, Yao, et al._  The Eleventh International Conference on Learning Representations. 2023. [[paper](https://openreview.net/pdf?id=0qSOodKmJaN)]
  - SLiC-HF: Sequence likelihood calibration with human feedback. _Zhao, Yao, et al._ arXiv preprint arXiv:2305.10425 (2023).[[paper](https://arxiv.org/pdf/2305.10425)][[huggingface](https://huggingface.co/papers/2305.10425)]
  - How can we know when language models know? on the calibration of language models for question answering. _Jiang, Zhengbao, et al._ Transactions of the Association for Computational Linguistics 9 (2021): 962-977.[[paper](https://aclanthology.org/2021.tacl-1.57.pdf)][[code](https://github.com/jzbjyb/lm-calibration)]
  - Calibrating deep neural networks using focal loss. _Mukhoti, Jishnu, et al._ Advances in Neural Information Processing Systems 33 (2020): 15288-15299.[[paper](https://arxiv.org/pdf/2002.09437)][[code](https://github.com/torrvision/focal_calibration)]

#### Post-hoc Calibration Methods

- **Temperature Scaling**
  - On calibration of modern neural networks. _Guo, Chuan, et al._ International conference on machine learning. PMLR 2017 [[paper](https://arxiv.org/pdf/1706.04599)][[code](https://github.com/gpleiss/temperature_scaling.git)]

- **Bayesian Binning into Quantiles (BBQ)**
  - Obtaining well calibrated probabilities using bayesian binning. _Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht._ Proceedings of the AAAI conference on artificial intelligence. Vol. 29. No. 1. 2015. [[paper](https://dl.acm.org/doi/10.5555/2888116.2888120)][[code](https://github.com/pakdaman/calibration)]

- **Scale-binning calibrator**
  - Verified uncertainty calibration. _Kumar, Ananya, Percy S. Liang, and Tengyu Ma._ Advances in Neural Information Processing Systems 32 (2019).[[paper](https://arxiv.org/pdf/1909.10155)][[code](https://github.com/p-lambda/verified_calibration)]

#### Closed-Box Calibration of Language Models

- **Auxiliary Model**
  - Calibrating Large Language Models Using Their Generations Only _Ulmer, Dennis, et al._ arXiv preprint arXiv:2403.05973 (2024). [[paper](https://arxiv.org/pdf/2403.05973)][[code](https://github.com/parameterlab/apricot/blob/main/README.md?plain=1)]
  - Llamas Know What GPTs Don't Show: Surrogate Models for Confidence Estimation. _Shrivastava, Vaishnavi, Percy Liang, and Ananya Kumar_  arXiv preprint arXiv:2311.08877 (2023).[[paper](https://arxiv.org/pdf/2311.08877)]

- **Linguistic Calibration**
  - Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. _Tian, Katherine, et al._  arXiv preprint arXiv:2305.14975 (2023).[[paper](https://arxiv.org/pdf/2305.14975)]
  - Navigating the grey area: How expressions of uncertainty and overconfidence affect language models. _Zhou, Kaitlyn, Dan Jurafsky, and Tatsunori Hashimoto_ arXiv preprint arXiv:2302.13439 (2023) [[paper](https://aclanthology.org/2023.emnlp-main.335.pdf)][[code](https://github.com/katezhou/navigating_the_grey/tree/main)]
  - Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. _Xiong, Miao, et al._  arXiv preprint arXiv:2306.13063 (2023) [[paper](https://arxiv.org/pdf/2306.13063)][[code](https://github.com/MiaoXiong2320/llm-uncertainty)]
  - Reducing conversational agents’ overconfidence through linguistic calibration. _Mielke, Sabrina J., et al._ Transactions of the Association for Computational Linguistics 10 (2022): 857-872. [[paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00494/112606/Reducing-Conversational-Agents-Overconfidence)][[data](https://parl.ai/projects/metacognition/)]
  - Teaching models to express their uncertainty in words. _Lin, Stephanie, Jacob Hilton, and Owain Evans._ arXiv preprint arXiv:2205.14334 (2022).[[paper](https://arxiv.org/pdf/2205.14334)][[code](https://github.com/sylinrl/calibratedmath)]


- **Self-Consistency**
  - Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms.  _Xiong, Miao, et al._ The Twelfth International Conference on Learning Representations (2024).[[paper](https://openreview.net/pdf?id=gjeQKFxFpZ)][[code](https://github.com/MiaoXiong2320/llm-uncertainty)]
  - Self-consistency improves chain of thought reasoning in language models. _Wang, Xuezhi, et al._ The Eleventh International Conference on Learning Representations (2023). [[paper](https://openreview.net/pdf?id=1PL1NIMMrw)][[code](https://github.com/codelion/optillm/blob/main/optillm/self_consistency.py)]

- **In-Context Learning**
  - Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering. _Zhou, Han, et al._ The Twelfth International Conference on Learning Representations. 2024. [[paper](https://arxiv.org/pdf/2309.17249)][[code](https://github.com/cambridgeltl/ClaPS/blob/main/algs/test_time_bn.py)]
  - Prototypical Calibration for Few-shot Learning of Language Models _Zhixiong Han and Yaru Hao and Li Dong and Yutao Sun and Furu Wei_ The Eleventh International Conference on Learning Representations. 2023.[[paper](https://openreview.net/forum?id=nUsP9lFADUF)][[code](https://github.com/ZihanWangKi/x-TC/blob/main/external/prompt_gpt/ProtoCal.py)]
  - Mitigating label biases for in-context learning. _Fei, Yu, et al._ Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023. [[paper](https://aclanthology.org/2023.acl-long.783.pdf)][[code](https://github.com/fywalter/label-bias)]
  - Answer-level calibration for free-form multiple choice question answering. _Kumar, Sawan._ Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022.[[paper](https://aclanthology.org/2022.acl-long.49.pdf)][[code](https://github.com/SawanKumar28/alc)]
  - Calibrate before use: Improving few-shot performance of language models. _Zhao, Zihao, et al._ International conference on machine learning. PMLR, 2021.[[paper](https://arxiv.org/pdf/2102.09690)][[code](https://github.com/tonyzhaozh/few-shot-learning/tree/main?tab=readme-ov-file)]

- **Sequence-level Calibration**
  -  Analyzing uncertainty in neural machine translation. _Ott, Myle, et al._ International Conference on Machine Learning. PMLR, 2018. [[paper](https://arxiv.org/pdf/1803.00047)][[code](https://github.com/facebookresearch/analyzing-uncertainty-nmt)]

#### Calibration in Domain Shift and Adaptation (DA)

- **TransCal**
  -  Transferable calibration with lower bias and variance in domain adaptation. _Wang, Ximei, et al._ Advances in Neural Information Processing Systems 33 (2020): 19212-19223. [[paper](https://papers.nips.cc/paper/2020/hash/df12ecd077efc8c23881028604dbb8cc-Abstract.html)][[code](https://github.com/thuml/TransCal)]

## Contributing

For contributions, please open an issue or submit a pull request.
