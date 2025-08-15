# DiagGym
<div align="center">
  <img src="./assets/logo.png" width="250"/>
  <div align="center"></div>
</div>


<p align="center">
          🤖 <a href="">Model</a>&nbsp&nbsp | &nbsp&nbsp 📊 <a href="">Data</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper Coming Soon</a> 
</p>

We introduce **DiagGym**, the first open‑source virtual clinical environment for training large language models (LLMs) as **diagnostic agents** with reinforcement learning (RL).  
DiagGym simulates realistic, multi‑turn diagnostic workflows by generating examination results conditioned on evolving patient states, enabling safe, closed‑loop training without real‑world risk.

Within DiagGym, we train **DiagAgent**—a family of RL‑optimized diagnostic agents (**7B**, **8B**, **14B**)—to actively manage diagnostic trajectories: selecting the most informative examinations, deciding when to stop, and committing to accurate final diagnoses.  

All models are available on Hugging Face for reproduction and extension.


## 🚀 Key Insights & Contributions

- 🏥 **First Open‑Source Diagnostic RL Gym** – **[DiagGym](https://huggingface.co)**: a high‑fidelity EHR world model that simulates examination outcomes for safe, interactive training and evaluation of diagnostic agents.  
- 🤗 **RL‑Trained Diagnostic Agents** – **[DiagAgent‑7B](https://huggingface.co)**, **[DiagAgent‑8B](https://huggingface.co)**, and **[DiagAgent‑14B](https://huggingface.co)**, trained in DiagGym, surpass 12 SOTA LLMs and prompt‑engineered agents in both single‑turn and end‑to‑end diagnostic tasks.  
- 🎯 **Closed‑Loop Learning Advantage** – RL in a realistic simulation yields up to **15.12%** higher diagnostic accuracy and **23.09%** higher examination recommendation F1 compared to the best baseline (including DeepSeek-v3, GPT-OSS-120B, and Claude-4).

<img src="assets/teaser.png"/> 


## Quick Start (Inference)

## Main Results

## Data Construction

## Model Training

## Evaluation


## 📝Citation & Contact

Our paper is comming soon ...

For any inquiries or feedback, don’t hesitate to contact henrychur@sjtu.edu.cn.
