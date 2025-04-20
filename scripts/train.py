import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_from_disk

# Load preprocessed dataset
dataset = load_dataset("json", data_files="data/formatted_dolly.json")

# Load tokenizer and base model
tokenizer = LlamaTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf")  # or a smaller version
model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")

# Tokenize dataset


def tokenize(batch):
    return tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=512)


tokenized_dataset = dataset.map(tokenize)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="models/llama-lora-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    logging_dir="logs",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="no",
    save_total_limit=2,
    fp16=True,  # if your GPU supports it
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Save model
model.save_pretrained("models/llama-lora-checkpoints")
tokenizer.save_pretrained("models/llama-lora-checkpoints")

print("✅ Training complete and model saved!")
