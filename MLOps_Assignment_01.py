
#!pip install torch
#!pip install transformers
#!pip install gradio

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Save the model locally
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained("saved_model")
model = BertForSequenceClassification.from_pretrained("saved_model")

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    sentiment = ["very negative", "negative", "neutral", "positive", "very positive"][predicted_class_id]
    return sentiment

# Create a Gradio interface
iface = gr.Interface(
    fn=classify_text,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis with BERT",
    description="Enter a text to get the sentiment classification using a pre-trained BERT model."
)

# Run the interface
iface.launch()

#loom recording:
#https://www.loom.com/share/8b29aff0172248f4970b91a10f331289