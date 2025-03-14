import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset("csv", data_files={
    "train": "fintech_train.csv",
    "validation": "fintech_val.csv",
    "test": "fintech_test.csv"
})

# Load tokenizer and model
model_name = "facebook/opt-1.3b"  # Choose a suitable LLM

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["Query"], padding="max_length", truncation=True, max_length=128)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",       # Model checkpoints
    logging_dir="./logs",         # Make sure this path exists
    logging_steps=50,             # Log every 50 steps
    report_to="tensorboard",      # Send logs to TensorBoard
    save_strategy="epoch",        # Save model checkpoints at the end of each epoch
)


# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()

# Save fine-tuned model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

print("Fine-tuning complete! Model saved in ./finetuned_model")
