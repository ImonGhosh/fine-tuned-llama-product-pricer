# Fine-Tuned LLaMA Product Pricer

## Goal
Build a robust product price prediction system for Amazon marketplace listings and compare traditional ML, frontier LLMs with fine-tuned Llama 3.1 8B open-source small language model.

## Process Overview
1. **Dataset Curation** (`01-dataset-curation-1.ipynb`, `02-dataset-curation-2.ipynb`)
   - Filtered Amazon product reviews from the [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) corpus.
   - Produced a clean training/evaluation corpus hosted at [`imonghose/pricer-data`](https://huggingface.co/datasets/imonghose/pricer-data).
2. **Baseline Modeling** (`03-baseline-models.ipynb`)
   - Benchmarked linear regression, bag-of-words, and Word2Vec models.
3. **Frontier Model Experiments** (`04-frontier_models.ipynb`, `05-fine_tuned_gpt.ipynb`)
   - Evaluated GPT-4o and GPT-4o-mini and attempted fine-tuning with OpenAI jobs.
4. **Open-Source LLaMA Pipeline** (`06-qlora_intro.ipynb`, `07-base_llama_evaluation.ipynb`)
   - Loaded the Meta-Llama-3.1-8B base model with 4-bit quantization and established evaluation baselines.
5. **QLoRA Fine-Tuning** (`08-fine_tune_llama_model.ipynb`)
   - Applied 4-bit quantization-aware LoRA (QLoRA) fine-tuning on 20K curated samples.
6. **Evaluation & Leaderboard** (`09_test_fine_tuned_llama.ipynb`, `10_leaderboard.ipynb`)
   - Compared all approaches on error (Avg $ Error, RMSLE) and accuracy metrics.

## Model Comparison Highlights
| Rank | Model | Avg Error ($) | RMSLE | Accuracy (%) |
| ---- | ----- | ------------- | ----- | ------------ |
| 1 | **Finetuned LLaMA 3.1 8B 4-bit (20K samples)** | **64.10** | **0.47** | **58.8** |
| 2 | GPT-4o | 76.42 | 0.86 | 56 |
| 3 | GPT-4o Mini | 89.73 | 0.83 | 47 |
| 4 | GPT-4o Mini (fine-tuned) | 91.45 | 0.68 | 44 |
| 5 | Word2Vec + SVM | 109.26 | 0.90 | 28.4 |
| 6 | BOW + Linear Regression | 113.6 | 0.99 | 24.8 |
| 7 | Word2Vec + Linear Regression | 115.14 | 1.05 | 23.6 |
| 8 | Human Predictions | 126.55 | 1.00 | 32 |

The fine-tuned 4-bit LLaMA 3.1 8B model clearly leads, outperforming GPT-4o and all baseline models on every evaluation metric.

## Usage & Artifacts
### Dataset
```python
from datasets import load_dataset

dataset = load_dataset("imonghose/pricer-data")
train = dataset["train"]
test = dataset["test"]
```

### Fine-Tuned Model (LoRA adapters)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = "imonghose/product-pricer-llama-2025-10-19_09.16.39"
REVISION = None  # set if you need a specific adapter snapshot
QUANT_4_BIT = True

if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_MODEL,
    revision=REVISION,
)

print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")
```

## Key Conclusions
- A well-targeted QLoRA fine-tuning run on an open-source 8B model delivers the best blend of accuracy and efficiency.
- GPT-4o fine-tuning yielded minimal gains, highlighting the importance of domain-specific fine-tuning data.
- Traditional ML baselines and human intuition lag significantly behind specialized LLM solutions.

## Future Directions
- Expand domain expertise with retrieval-augmented generation (RAG) over structured product catalogs.
- Scale training data beyond 20K samples and explore curriculum or multi-task fine-tuning.
- Evaluate latency/cost trade-offs by deploying the 4-bit model via optimized serving stacks (e.g., vLLM or TGI).

## Repository Structure
- `01-10*.ipynb`: Sequential experimentation notebooks covering curation, baselines, frontier models, QLoRA fine-tuning, evaluation, and leaderboard analysis.
- `wandb/`: Tracking artifacts for hyperparameters and experiment metrics.

