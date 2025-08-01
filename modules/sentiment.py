from transformers import pipeline

# Initialize once
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["Strong Negative", "Mild Negative", "Actionable", "Unclear"]

def classify_sentiment(text):
    result = classifier(text, candidate_labels=labels)
    return result["labels"][0]
