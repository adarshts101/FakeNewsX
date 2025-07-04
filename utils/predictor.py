# utils/predictor.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model using safetensors
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float32
).to(device)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "Real 🟢" if pred == 1 else "Fake 🔴"
    confidence = float(probs[0][pred])
    return label, confidence

def predict_proba(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()[0]
