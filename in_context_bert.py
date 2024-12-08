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

## Python Function to help in the intent_classification based on a given prompt
def classify_intent(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # print(logits)
    # print(logits.size())
    predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label


in_context_zero_shot = """I want you to help me classify the intent of a given statement based on the mapping given below
We have 3 types of intents where:
0 -> Book a Cab
1 -> Get me the Weather
2 -> Book a Restaurant Table

Based on this mapping help me I want you to give me the Label for a given input sentence.
"""

in_context_15_shot = """I want you to help me classify the intent of a given statement based on some examples given below
We have 3 types of intents where:
0 -> Book a Cab
1 -> Get me the Weather
2 -> Book a Restaurant Table

Example 1:
Input Sentence - "Find me a ride to CMU"
Label - 0

Example 2:
Input Sentence - "Get me a cab from Target to CMU for 2 people."
Label - 0

Example 3:
Input Sentence - "Arrange an Uber to Masala House."
Label - 0

Example 4:
Input Sentence - "Book me an Uber to Downtown Pittsburgh."
Label - 0

Example 5:
Input Sentence - "I need an Uber to East Liberty for 5 at 07:45 PM."
Label - 0


Example 6:
Input Sentence - "Can you reserve a table at Chipotle, Forbes Avenue."
Label - 2

Example 7:
Input Sentence - "I need a table at Stack'd at 10pm"
Label - 2

Example 8:
Input Sentence - "Can you help me reserve a table at Reva for 6 people at 8pm."
Label - 2

Example 9:
Input Sentence - "I need a table at Stack'd at 10pm"
Label - 2

Example 10:
Input Sentence - "Figure out if I can get a reservation at Ma Peche after the concert"
Label - 2

Example 11:
Input Sentence - "How is the weather outside?"
Label - 1

Example 12:
Input Sentence - "Will it rain in Pittsburgh tomorrow?"
Label - 1

Example 13:
Input Sentence - "Is it sunny in Kolkata."
Label - 1

Example 14:
Input Sentence - "How's the weather today in Seattle."
Label - 1

Example 15:
Input Sentence - "Show me the weather conditions in Mumbai."
Label - 1


End of Examples.

Now learning from the above examples, I want you to give me the Label for a given input sentence.
"""


# Load tokenizer and model, move to GPU
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model_for_in_context = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels = 3).to(device)

input_sent = "I need a cab to CMU"
prompt = generate_prompt_for_intent_classification(in_context_15_shot, input_sent)
print(prompt)

# Classify using Flan-T5
predicted_label_in_context = classify_intent(prompt, tokenizer, model_for_in_context)
print(f"Predicted Label: {predicted_label_in_context}")

pred_label_in_context = []
for query in df_in_context["text"]:
    prompt = generate_prompt_for_intent_classification(in_context_15_shot, query)
    pred_label_in_context.append(classify_intent(prompt, tokenizer, model_for_in_context))

df_in_context["predicted_labels"] = pred_label_in_context
print(df_in_context.predicted_labels.value_counts())

incontext_accuracy = (df_in_context.label == df_in_context.predicted_labels).mean()
print(f"In Context 15-Shot Accuracy with BERT: {incontext_accuracy}")