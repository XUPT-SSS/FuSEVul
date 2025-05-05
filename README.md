# Enhancing Vulnerability Detection by Fusing Code Semantic Features with LLM-Generated Interpretations
The rising prevalence of software vulnerabilities highlights the critical need for more effective detection techniques. Although recent deep learning (DL) approaches have made notable progress, they primarily rely on code-centric modalities, such as token sequences, abstract syntax trees (ASTs), and graph-based representations, while largely neglecting complementary semantic cues available from natural language artifacts like code explanations. This paper proposes FuSEVul, a novel multi-modal framework that integrates code semantics with automatically generated natural language explanations to enhance vulnerability detection. FuSEVul consists of three key components: (1) a code semantic encoder that leverages pre-trained model to capture structural and semantic features from source code; (2) an explanation generation module that prompts a large language model (LLM) to produce functional and risk-aware textual descriptions, subsequently encoded using RoBERTa; and (3) a self-attention-based fusion mechanism that dynamically aligns and integrates cross-modal features, emphasizing signals most indicative of vulnerabilities. Extensive experimental evaluations across three public datasets demonstrate that FuSEVul outperforms 18 state-of-the-art baselines, yielding average relative gains of 7.3% in accuracy and 52.2% in F1 score. Ablation studies further confirm the effectiveness of incorporating LLM-generated explanations and the proposed fusion strategy.

# Design of FuSEVul
<div align="center">
  <img src="https://github.com/XUPT-SSS/FuSEVul/blob/main/frame.jpg">
</div>

# Datasets

# Source
## Step1:Code normalization
```
cd Normalization
python normalization.py
```
## Step2:Code Explanation Generation 
```
cd model
python comment.py
```
## Step3:Train models
```
python run.py
```
