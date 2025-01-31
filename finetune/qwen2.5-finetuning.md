**@qwen2.5-finetuning.md**

# Fine-tuning Qwen2.5 (1.5B) in a Standard Python Environment with ROCm Support

This guide provides a step-by-step procedure to fine-tune the Qwen2.5 \(1.5B\) model on a custom instruction-following task (or any other text generation dataset) using LoRA and 4-bit quantization. These instructions assume you are running inside a Docker container built from the provided ROCm 6.1.2 Dockerfile, which already has:

1. **ROCm toolchain** (HSA, hip, rocm-device-libs)
2. **PyTorch** installed with ROCm 6.1 support
3. **bitsandbytes** compiled for gfx1030
4. **Basic build tools** (cmake, git, python3, etc.)

Below, you will find:

- [Environment Setup](#environment-setup)
- [Verifying bitsandbytes Installation](#verify-bitsandbytes)
- [Data Preparation](#data-preparation)
- [Model Preparation (4-bit + LoRA)](#model-preparation)
- [Fine-tuning Script](#fine-tuning-script)
- [Optional: Evaluation & Inference](#evaluation-and-inference)
- [Saving & Loading the Fine-tuned Model](#saving-and-loading)

---

## 1. Environment Setup <a name="environment-setup"></a>

Inside your Docker container, install the necessary Python libraries not included by default. We will need:

- **Transformers** (for model loading & tokenization)
- **Accelerate** (for multi-GPU / efficient training)
- **PEFT** (for LoRA-based parameter-efficient fine-tuning)
- **Datasets** (for preparing custom data)
- **TRL** (for advanced training scripts, e.g. DPO, PPO, or SFT)

```bash
root@container:/workspace# pip install \
  accelerate \
  transformers \
  peft \
  datasets \
  trl
```

*(You may also wish to install `safetensors` if dealing with large model checkpoint files.)*

---

## 2. Verifying bitsandbytes Installation <a name="verify-bitsandbytes"></a>

To confirm bitsandbytes is correctly compiled and using ROCm acceleration, run a quick check in Python:

```bash
root@container:/workspace# python
```

```python
import bitsandbytes as bnb
print("bitsandbytes version:", bnb.__version__)

# Attempt to load the CUDA (ROCm) setup:
print("bitsandbytes CUDA/ROCm check:", bnb.cuda_setup.main_check())
```

You should see a message indicating that ROCm is recognized and your GPU architecture (gfx1030) is being used.

---

## 3. Data Preparation <a name="data-preparation"></a>

You will need a dataset of **(instruction, response)** examples or some text corpus for language modeling. For demonstration, we will assume we have a local dataset or a Hugging Face dataset.  
If your dataset is on Hugging Face Hub, you can fetch it with:

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/helpful-instructions", split="train")
print("Number of samples:", len(dataset))
```

**Converting dataset to the correct format:**  
To fine-tune an instruction model, we typically transform each row into a single prompt of the form:

```
<human>: {instruction}
<assistant>: {response}
```

For example:

```python
def format_instruction(ex):
    prompt_text = f"<human>: {ex['instruction']}\n<assistant>: {ex['demonstration']}"
    return {"text": prompt_text}

dataset = dataset.map(format_instruction)
```

Once transformed, you can tokenize:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token = tokenizer.eos_token  # ensure there's a pad token

def tokenize_function(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["instruction", "demonstration", "text", ...])
tokenized_dataset.set_format("torch")
```

> **Note:** Adjust `max_length` and `batch_size` according to GPU memory.

---

## 4. Model Preparation (4-bit + LoRA) <a name="model-preparation"></a>

### a) Load Qwen 2.5 (1.5B) in 4-bit

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
```

### b) Apply LoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare the 4-bit loaded model for LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  # for causal language modeling
)

model = get_peft_model(model, lora_config)

# Quick optional function to verify how many parameters are trainable
def print_trainable_params(m):
    trainable = 0
    total = 0
    for name, param in m.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable params: {trainable} / {total} = {100*trainable/total:.2f}%")

print_trainable_params(model)
```

---

## 5. Fine-tuning Script <a name="fine-tuning-script"></a>

Below is a simple training script using **Hugging Face Transformers** standard `Trainer`. Create a Python file (e.g. `finetune_qwen.py`) with the following content:

```python
#!/usr/bin/env python

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main():
    # 1. Load and format dataset
    dataset = load_dataset("HuggingFaceH4/helpful-instructions", split="train")

    def format_instruction(ex):
        prompt_text = f"<human>: {ex['instruction']}\n<assistant>: {ex['demonstration']}"
        return {"text": prompt_text}

    dataset = dataset.map(format_instruction)

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512, padding="max_length")

    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("torch")

    # 3. Load Qwen model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. Prepare LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    # 5. TrainingArgs & Trainer
    training_args = TrainingArguments(
        output_dir="./qwen_finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_steps=200,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="none",  # or "tensorboard"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 6. Train
    model.config.use_cache = False  # for gradient checkpointing
    trainer.train()

    # 7. Save final model & tokenizer
    model.save_pretrained("./qwen_finetuned")
    tokenizer.save_pretrained("./qwen_finetuned")

if __name__ == "__main__":
    main()
```

Then run:

```bash
root@container:/workspace# python finetune_qwen.py
```

You should see training logs indicating steps, losses, and final checkpoint saves in `./qwen_finetuned`.

---

## 6. Optional: Evaluation & Inference <a name="evaluation-and-inference"></a>

After training completes, you can test your model interactively:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./qwen_finetuned")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")
model = PeftModel.from_pretrained(base_model, "./qwen_finetuned")
model.to(device)

prompt = "<human>: What equipment do I need for rock climbing?\n<assistant>:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 7. Saving & Loading the Fine-tuned Model <a name="saving-and-loading"></a>

- **Saving** is covered above (`model.save_pretrained`, `tokenizer.save_pretrained`).
- **Loading** your LoRA-adapted Qwen model later:

  ```python
  from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer

  base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", device_map="auto")
  lora_model = PeftModel.from_pretrained(base_model, "./qwen_finetuned")
  tokenizer = AutoTokenizer.from_pretrained("./qwen_finetuned")
  ```

That’s all! You now have a parameter-efficient Qwen2.5 model fine-tuned for your custom instructions using LoRA with 4-bit quantization—ensuring your GPU memory usage remains manageable on ROCm hardware.

---

## Summary

1. **Install** extra libraries (transformers, accelerate, peft, datasets, trl).
2. **Verify** bitsandbytes is using ROCm / gfx1030.
3. **Prepare** your dataset → format → tokenize.
4. **Load** Qwen 2.5 in 4-bit + **LoRA**.
5. **Train** with `Trainer` or your preferred library (e.g. `SFTTrainer` from `trl`).
6. **Save** model & tokenizer for inference.
7. (Optionally) **Evaluate** or do **inference** with the new LoRA checkpoint.

Enjoy fine-tuning Qwen on ROCm!