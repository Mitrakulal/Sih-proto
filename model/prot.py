# train_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

class LogDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

# Load data
print("Loading tokenized data...")
with open("C:\\Users\\kulal\\Desktop\\tryproto\\tocken\\tokens.txt", "r") as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(texts)} samples")

# Initialize tokenizer and model
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Create dataset
dataset = LogDataset(texts, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./log_model',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=50,
    logging_steps=10,
    remove_unused_columns=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
trainer.save_model()
tokenizer.save_pretrained('./log_model')

print("Training complete! Model saved to ./log_model")