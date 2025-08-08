from transformers import pipeline

labels = ["Strong Negative", "Mild Negative", "Actionable", "Unclear"]

# Initialize once
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_sentiment(text):
    # Ensure text is valid
    if not text or not str(text).strip():
        return "Unclear"  # default for empty responses
    
    result = classifier(str(text), candidate_labels=labels)
    return result["labels"][0]
