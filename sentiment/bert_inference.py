#imports
from datasets import load_dataset
import numpy as np
import pandas as pd
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

print("################################## Running Infernece for BERT ################################")

# checkpoint number needs to be set every time we have different value for it.
ckpt_no = "158"
model_name="bert"

# Loading test data
df_test = pd.read_csv("../data/sentiment/splits/test.csv")
test_source = df_test["tweet"].tolist()
test_target = df_test["label"].tolist()


print('\n\nLoading Model.......') # Uncoment the required model on which you want to perform evaluation.


# Loading Pretrained Model
# tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
# model = AutoModelForSequenceClassification.from_pretrained("bert-large-cased", num_labels=2)

# Loading baseline finetuned model

# Loading DPS finetuned model
model_checkpoint = "../DPS/script/trained_models/output_"+model_name+"/checkpoint-"+ckpt_no
model = AutoModelForSequenceClassification.from_pretrained("../DPS/script/trained_models/output_"+model_name+"/checkpoint-"+ckpt_no)
tokenizer = AutoTokenizer.from_pretrained("../DPS/script/trained_models/output_"+model_name+"/checkpoint-"+ckpt_no)

# test_text = "I went to agra yesterday what a pleasent day it was."

# model_inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

print("\n\nEvaluation started.......")

predictions = []

for e in test_source:
    inputs = tokenizer(e, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    pred = model.config.id2label[predicted_class_id]
    if pred == "LABEL_1":
        predictions.append("positive")
    else:
        predictions.append("negative")


print("\n\nCalculating accuracy.......")

true_positives = 0
for i in range(len(test_target)):
    if test_target[i] == predictions[i]:
        true_positives+=1

precision = true_positives/len(predictions)
print("precision =",precision)

recall = true_positives/len(test_target)
print("recall =",recall)

f1_score = (2*precision*recall/(precision+recall))
print("F1-Score =",f1_score)