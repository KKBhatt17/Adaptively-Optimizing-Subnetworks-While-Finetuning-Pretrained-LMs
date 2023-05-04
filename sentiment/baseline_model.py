#imports

from datasets import load_dataset
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

print("\n\nLoading Dataset..............................")

dataset = load_dataset('csv', data_files={'train': '../data/sentiment/splits_int/train.csv', 'validation': '../data/sentiment/splits_int/val.csv' , 'test': '../data/sentiment/splits_int/test.csv'})

print(dataset)

tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')

# def tokenize_data(example):
#     return tokenizer(example['tweet'], padding='max_length')
def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
# dataset = dataset.map(transform_labels, remove_columns=['label'])


from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


# Training

print("\n\nStarting Training Pipeline...............")

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-large-cased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="finetuned_models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\n\nTraining Started..............................")

print(trainer.train())



# Code for evaluation.

# metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# trainer.evaluate()