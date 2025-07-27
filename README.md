# DoctorGemma-3
### Gemma-3 4B LLM Fine Tuned on 100K Doctor-Patient QA Dataset

<p align="left">
  <a href="https://huggingface.co">
    <img src="https://img.shields.io/badge/HuggingFace-FFD21F?logo=huggingface&logoColor=black" alt="Hugging Face"/>
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  </a>
  <a href="https://github.com/unslothai/unsloth">
    <img src="https://img.shields.io/badge/Unsloth-4B0082?logo=github&logoColor=white" alt="Unsloth"/>
  </a>
</p>


This project demonstrates how to fine-tune the `unsloth/gemma-3-4b-it` language model using Low-Rank Adaptation (LoRA) with 4-bit quantization. The setup is optimized for fast iteration, efficient memory use, and good generalization performance for instruction-based tasks.

Base Model - [Gemma-3 4B](https://huggingface.co/unsloth/gemma-3-4b-it)  
Dataset - [HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)


---

## 📦 Model Setup

```python
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)
```

## 🧩 LoRA Configuration
```python
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)
```

## 🏋️ Training Configuration
```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)
```

