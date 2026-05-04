<div align="center">

# ERGO-Extended: Inference-Time Multi-Turn Context Consolidation
<br>

<!-- ![ERGO Banner](READMEimg/Representative_Diagram.png) -->

<!-- [![Paper](https://img.shields.io/badge/📄_Read_Paper-8A2BE2?style=for-the-badge)](https://ergopaper.github.io/ERGO/static/pdfs/ERGO.pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

![Tests](https://github.com/haziq-exe/ERGO/actions/workflows/tests.yml/badge.svg)

• [Paper Website](https://ergopaper.github.io/ERGO/) • [Quick Start](https://github.com/haziq-exe/ERGO?tab=readme-ov-file#quick-start) • [Repository Structure](https://github.com/haziq-exe/ERGO?tab=readme-ov-file#repository-structure) • [Results](https://github.com/haziq-exe/ERGO?tab=readme-ov-file#key-results) • [Contact](https://github.com/haziq-exe/ERGO?tab=readme-ov-file#contact) • -->

</div>

---

## Overview

Multi-turn LLMs show great potential in a wide range of domains, and increasingly in long-context deployment. However, they are vulnerable to errors, often caused in part by degraded context, which then propagate throughout a conversation. Existing work reduces the occurrence of errors by giving feedback, retrieving a specific piece of context or external data, or resetting context, but these solutions often lack robustness or may lead to more errors. We propose to improve upon existing work in error correction by detecting more signals of potential for hallucination, consolidating degraded multi-turn context, and using safeguards to ensure that the model is not distracted by irrelevant information and does not hallucinate misinformation. We present a combination of existing methods for utilizing context effectively and efficiently so that the model can provide an appropriate response given the context while minimizing additional cost, latency, complexity, information loss, and risk of hallucination.

We build upon **ERGO** for inference-time context rewriting, extending it by detecting three signals to trigger context consolidation: Shannon entropy, probability, and perplexity.

<!-- ## Core Results

<div align="center">
<table style="table-layout: fixed; width: 100%;">
<tr>
<td align="center" style="white-space: nowrap; width: 33%;">
<h3>56.6%</h3>
<b>Average Performance Gain</b>
</td>
<td align="center" style="white-space: nowrap; width: 33%;">
<h3>24.7%</h3>
<b>Peak Capability Increase</b>
</td>
<td align="center" style="white-space: nowrap; width: 33%;">
<h3>35.3%</h3>
<b>Decrease in Unreliability</b>
</td>
</tr>
</table>
</div> -->


## Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/RETprojects/ERGO-Extended.git
cd ERGO-Extended
pip install -r requirements.txt
```

- To use OpenAI models you will need the environment variable "OPENAI_KEY" to be set to your key.

- You will need to downloaded the following [sharded dataset from Laban et al](https://huggingface.co/datasets/microsoft/lost_in_conversation)

### Basic Usage

```python
from experiments.runExperiment import RunExperiment

# Initialize experiment with your chosen model
experiment = RunExperiment(
    model_name="HuggingFaceTB/SmolLM-135M-Instruct",
    device="cpu",
    device_map=None,
    max_new_tokens=1000
)

# Run ERGO-Extended on GSM8K dataset
experiment.run_GSM8K(
    dataset_path="sharded_dataset.json", # path to sharded dataset from Laban et al.
    num_Qs=20,
    num_runs=1,
    threshold_H=0.03, 
    threshold_p=-0.1, 
    threshold_PPL=50, 
    output_path="outputs/gsm8k_example.json"
)
```

Run from root directory:
```bash
python -m main.example_main
```

## Repository Structure

```
ERGO-Extended/
│
├── evaluation/         # Evaluation metrics and scoring
│   └── evaluator.py
|   └── utils.py 
|   └── eval.bfcl.py    # Taken from Laban et al.
│
├── core/               # Core ERGO-Extended implementation
│   ├── dataset.py         
│   ├── model.py          
│   └── utils.py          
│
├── experiments/        # Experiment runner
│   └── runExperiment.py  
│
├── generation/         # Generate with ERGO-Extended
│   └── generator.py
│
└── main/              # Example scripts
    └── example_main.py
```

## Evaluated Tasks

ERGO-Extended has been rigorously tested across three diverse generation tasks:

<div align="center">

| Task | Dataset | Description | Metric |
|:------:|:---------:|:-------------:|:--------:|
| **Math** | GSM8K | Elementary math word problems | Exact Match |
| **Code** | LiveCodeBench | Python function generation | Test Suite Pass |
| **API Calls** | Berkeley FCL | Function calling from instructions | Call Validity |

</div>

<!-- ## Key Results

<div align="center">

### Average Performance Across Models

| Model | ERGO | ERGO-Extended | **Relative Improvement** |
|:-------:|:------:|:---------:|:----------:|:-------------:|
| GPT-4.1-mini | 84.7 | 74.1 | **-10.6%** | 
| LLaMA-3.1-8B | 44.3 | 23.5 | **-20.8%** |

</div> -->

## Citation

If you use ERGO-Extended in your research, please cite our paper:

```bibtex
@misc{toutin-etal-2026-ergo-extended,
    title = "Effective Context Utilization for Multi-Turn LLMs",
    author = "Toutin, Rémi  and
      Madisetti, Vijay K.",
}
```

## Contact

**Lead Author**: Rémi Toutin  
📧 rtoutin3@gatech.edu

**Corresponding Author**: Dr. Vijay K. Madisetti  
📧 vkm@gatech.edu


## Code References

- **ERGO (Khalid et al)** — code accompanying the paper *ERGO: Entropy-guided Resetting for Generation Optimization*  
https://github.com/haziq-exe/ERGO
- **Lost in Conversation (Laban et al)** — code accompanying the paper *LLMs Get Lost in Multi-Turn Conversation*  
https://github.com/microsoft/lost_in_conversation

<!-- 
---

[Back to Top](https://github.com/RETprojects/ERGO-Extended?tab=readme-ov-file#ergo-entropy-guided-resetting-for-generation-optimization) -->

</div>