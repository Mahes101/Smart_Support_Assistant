# nlp_models.py
import spacy
from spacy.cli import download as spacy_download
from transformers import pipeline

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# We'll cache / reuse the sentiment pipeline instead of recreating it every call
_sentiment_pipeline = None

def get_sentiment_pipeline(device: int = -1):
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        # Choose device argument: -1 means CPU, >=0 means GPU index
        # Avoid letting pipeline choose device_map="auto" which can trigger meta-tensor issues
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",  # or whichever model you prefer
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )
    return _sentiment_pipeline

def analyze_text(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Use the cached pipeline
    sentiment_pipe = get_sentiment_pipeline(device=-1)  # device = -1 (CPU) or 0 (GPU) etc
    result = sentiment_pipe(text)  # returns a list of dicts
    sentiment = result[0] if result else None

    return {
        "entities": entities,
        "sentiment": sentiment
    }
