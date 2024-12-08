import pandas as pd
import os, requests, json, re
import requests
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
from collections import Counter


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Read the CSV
df = pd.read_csv("synthetic_intent_dataset_v2.csv")

df.loc[df["label"] == 1, "label"] = 0
df.loc[df["label"] == 2, "label"] = 1
df.loc[df["label"] == 7, "label"] = 2
df_in_context = df.copy()

# Python Function to generate Few Shot Classification Prompt
def generate_prompt_for_intent_classification(verbalizer, input_sentence) -> str:
    return f"{verbalizer}\nInput Sentence: {input_sentence}\nOutput:"

in_context_zero_shot = """I want you to help me classify the intent of a given statement based on the mapping given below
We have 3 types of intents where:
0 -> Book a Cab
1 -> Get me the Weather
2 -> Book a Restaurant Table
...

Based on this mapping help me I want you to give me the Label for a given input sentence.
"""

in_context_few_shot = """I want you to help me classify the intent of a given statement based on some examples given below
We have 3 types of intents where:
0 -> Book a Cab
1 -> Get me the Weather
2 -> Book a Restaurant Table

Example 1:
Input Sentence - "Find me a ride to CMU"
Label - 0

Example 2:
Input Sentence - "Can you reserve a table at Chipotle, Forbes Avenue."
Label - 2

Example 3:
Input Sentence - "Will it rain in Pittsburgh tomorrow?"
Label - 1

End of Examples.

Now learning from the above examples, I want you to give me the Label for a given input sentence.
"""

# Load tokenizer and model, move to GPU
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

# Function to classify intent
def classify_intent(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10)
    
    # Decode the output tokens to text
    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return predicted_label

input_sent = "I want to go to Reva."
prompt = generate_prompt_for_intent_classification(in_context_few_shot, input_sent)
print(prompt)

# Classify using Flan-T5
predicted_label_in_context = classify_intent(prompt, tokenizer, model)
print(f"Predicted Label: {predicted_label_in_context}")

pred_label_in_context = []
for query in df_in_context["text"]:
    prompt = generate_prompt_for_intent_classification(in_context_few_shot, query)
    pred_label_in_context.append(classify_intent(prompt, tokenizer, model))

df_in_context["predicted_labels"] = pred_label_in_context
print(df_in_context.predicted_labels.value_counts())

# Convert string labels back to numeric for comparison
df_in_context["predicted_labels"] = df_in_context["predicted_labels"].apply(
    lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else -1
)

incontext_accuracy = (df_in_context.label == df_in_context.predicted_labels).mean()
print(f"In Context 3-shot Accuracy: {incontext_accuracy}")