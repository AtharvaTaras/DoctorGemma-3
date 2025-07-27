# DoctorGemma-3
### Gemma-3 4B LLM Fine Tuned on 100K Doctor-Patient QA Dataset


This project demonstrates how to fine-tune the `unsloth/gemma-3-4b-it` language model using Low-Rank Adaptation (LoRA) with 4-bit quantization. The setup is optimized for fast iteration, efficient memory use, and good generalization performance for instruction-based tasks.

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

