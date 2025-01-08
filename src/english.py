from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import json
import numpy as np
import evaluate
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Model and tokenizer setup
model_checkpoint = "bert-base-uncased"

id2label = {0: "No", 1: "Yes"}
label2id = {"No": 0, "Yes": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Load dataset
with open("PromiseEval_Trainset_English.json", 'r') as file:
    data = json.load(file)

# Preprocess dataset
def preprocess_data(example):
    return {
        "text": example["data"],
        "label": label2id[example["promise_status"]],
    }

dataset = Dataset.from_list([preprocess_data(item) for item in data])

# Split dataset into train and test
train_test_split_ratio = 0.9
dataset = dataset.train_test_split(test_size=1-train_test_split_ratio, seed=42)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    return acc

# Training setup
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    evaluation_strategy="epoch",     # evaluate at the end of each epoch
    save_strategy="epoch",           # save the model at the end of each epoch
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    load_best_model_at_end=True,     # load the best model at the end
)

trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=tokenized_dataset['train'],   # training dataset
    eval_dataset=tokenized_dataset['test'],    # evaluation dataset
    data_collator=data_collator,         # data collator
    compute_metrics=compute_metrics      # metric computation
)

# Train the model
trainer.train()

# Evaluation
results = trainer.evaluate()

print("Evaluation Results:", results)

# Inference on the test set
test_dataset = tokenized_dataset['test']

test_dataloader = DataLoader(
    test_dataset.with_format("torch"),
    batch_size=8,
    collate_fn=data_collator
)

with torch.no_grad():
    logits_list = []
    labels_list = []
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits_list.append(logits)
        labels_list.append(labels)

# Concatenate logits and labels
logits = torch.cat(logits_list, dim=0)
labels = torch.cat(labels_list, dim=0)

# Get predictions
predictions = torch.argmax(logits, dim=-1).tolist()
true_labels = labels.tolist()

# Map predictions back to labels
predicted_labels = [id2label[pred] for pred in predictions]
true_labels_mapped = [id2label[label] for label in true_labels]

# Print results for each example
for i in range(len(test_dataset)):
    print(f"Text: {test_dataset[i]['text']}")
    print(f"Predicted: {predicted_labels[i]}, True: {true_labels_mapped[i]}")
    print("-" * 50)

# Compute overall accuracy on the test set
accuracy_result = compute_metrics((logits.cpu().numpy(), labels.cpu().numpy()))
print("Overall Accuracy:", accuracy_result)