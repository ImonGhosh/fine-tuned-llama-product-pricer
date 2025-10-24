import modal
from modal import App, Volume, Image
# Setup - define our infrastructure with code!

app = modal.App("product-pricer-llama-service")
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft")

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "hf-secret" with "huggingface-secret"
secrets = [modal.Secret.from_name("hf-secret")]

# Constants
GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "product-pricer-llama"
HF_USER = "imonghose" # your HF name here! Or use mine if you just want to reproduce my results.
RUN_NAME = "2025-10-19_09.16.39"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = None
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
# MIN_CONTAINERS = 0
MIN_CONTAINERS = 1
CACHE_DIR = "/root/.cache/huggingface/hub/"

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets, 
    gpu=GPU, 
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume}
)
class Pricer:

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
        
        # Quant Config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            quantization_config=quant_config,
            device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_MODEL, revision=REVISION)

    @modal.method()
    def price(self, description: str) -> float:
        import os
        import re
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
    
        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = self.fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
        result = self.tokenizer.decode(outputs[0])
    
        contents = result.split("Price is $")[1]
        contents = contents.replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0
    





# Code Explanation :
"""
@app.cls(...)
Turns the Python class into a Modal class (a stateful serverless worker with lifecycle).
The kwargs define the runtime for all its methods:
image=image.env({"HF_HUB_CACHE": CACHE_DIR}) → container image + an env var for Hugging Face cache path.
secrets=secrets → injects credentials at runtime.
gpu=GPU → requests a GPU-backed container.
timeout=1800 → each method call can run up to 30 minutes.
min_containers=MIN_CONTAINERS → keeps that many warm replicas ready (lower latency).
volumes={CACHE_DIR: hf_cache_volume} → mounts a persistent Modal Volume at CACHE_DIR so model files cached from HF survive across runs/containers.

You typically call it with Pricer().price.remote(...). Modal spins a container, attaches the volume, sets env, etc.

@modal.enter()
A lifecycle hook that runs once per container startup (per cold start), before any method is called.
Perfect for heavyweight setup you want to cache in memory: loading tokenizer/model, moving to GPU, setting quantization, etc.
Anything attached to self here (e.g., self.tokenizer, self.fine_tuned_model) is reused by later @modal.method calls in the same warm container, saving load time.

@modal.method()
Exposes an instance method as a remote-callable endpoint that runs inside the Modal container configured by @app.cls.
Gets access to the state initialized in @modal.enter() via self.
This is what you call with .remote(...) (or .map(...) for batches). In your snippet, price():
Builds the prompt, uses self.tokenizer and self.fine_tuned_model (already on GPU) to generate,
Parses the numeric price from the model’s output,
Returns a Python float.


Quick mental model
@app.cls → define the machine and its persistent mounts/env; it’s your “service”.
@modal.enter → boot the machine once and load big stuff (models) into memory/GPU.
@modal.method → handle requests using that preloaded state.

(And that Volume.from_name(..., create_if_missing=True) + HF_HUB_CACHE mapping is what makes HF model downloads stick around across containers, so cold starts don’t have to pull gigabytes again.)
"""