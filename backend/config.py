"""
Configuration file for the Sentiment Analyzer application
"""

import os
from pathlib import Path

# Project root directory - handle both local and HF Space environments
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths - with fallback for HuggingFace Spaces
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

# Log configuration for debugging
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Config BASE_DIR: {BASE_DIR}")
logger.debug(f"Config MODELS_DIR: {MODELS_DIR}")

# Data paths
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_PATH = DATA_DIR / "Reviews.csv"
TEST_DATA_PATH = DATA_DIR / "Test.csv"
OUTPUT_PATH = DATA_DIR / "Result.csv"

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
    "https://siva0801-sentimentanalysis-api.hf.space",
    "https://*.hf.space",
]

# API settings
API_TITLE = "Sentiment Analyzer API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "REST API for analyzing sentiment in product reviews"

# Model hyperparameters
MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Sentiment labels
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive"
}

SENTIMENT_EMOJI = {
    "Positive": "😊",
    "Negative": "😞"
}

# Text preprocessing settings
LOWERCASE = True
REMOVE_NUMBERS = True
REMOVE_PUNCTUATION = True
REMOVE_STOPWORDS = True
STOPWORDS_LANGUAGE = 'english'

# CSV export settings
EXPORT_COLUMNS = [
    'ProductId',
    'UserId',
    'Text',
    'Cleaned_Text',
    'Predicted_Sentiment',
    'Sentiment_Label',
    'Confidence_Score'
]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
