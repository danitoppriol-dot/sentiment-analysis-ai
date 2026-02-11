import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model
model = DistilBertForSequenceClassification.from_pretrained("./sentiment_model")
tokenizer = DistilBertTokenizer.from_pretrained("./sentiment_model")

model.eval()

while True:
    text = input("Enter a sentence (or 'exit'): ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    if prediction == 1:
        print("Positive")
    else:
        print("Negative")
