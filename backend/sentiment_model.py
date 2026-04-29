"""
Sentiment Model Training and Prediction
"""

import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, Any, Optional

from backend.config import (
    MODEL_PATH, TFIDF_PATH, TRAIN_DATA_PATH,
    MAX_FEATURES, TEST_SIZE, RANDOM_STATE, SENTIMENT_LABELS
)
from backend.preprocess import TextPreprocessor

logger = logging.getLogger(__name__)


class SentimentModel:
    """
    Sentiment Analysis Model using Logistic Regression and TF-IDF vectorization.
    
    This class handles:
    - Model training from raw reviews
    - Text preprocessing
    - Sentiment prediction
    - Model persistence (saving/loading)
    - Confidence score calculation
    """
    
    def __init__(self):
        """Initialize the SentimentModel."""
        self.model = None
        self.tfidf = None
        self.preprocessor = TextPreprocessor()
        self.is_trained = False
    
    def load_pretrained_model(self) -> bool:
        """
        Load pre-trained model and TF-IDF vectorizer from disk.
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            logger.info(f"Attempting to load model from {MODEL_PATH}")
            logger.info(f"Attempting to load vectorizer from {TFIDF_PATH}")
            logger.info(f"Model path exists: {MODEL_PATH.exists()}")
            logger.info(f"Vectorizer path exists: {TFIDF_PATH.exists()}")
            logger.info(f"MODEL_PATH absolute: {MODEL_PATH.absolute()}")
            logger.info(f"TFIDF_PATH absolute: {TFIDF_PATH.absolute()}")
            
            # Log dependency versions for debugging pickle compatibility
            import sklearn
            logger.info(f"scikit-learn version: {sklearn.__version__}")
            logger.info(f"numpy version: {np.__version__}")
            logger.info(f"joblib version: {joblib.__version__}")
            
            # Try loading from configured paths
            if MODEL_PATH.exists() and TFIDF_PATH.exists():
                try:
                    self.tfidf = joblib.load(TFIDF_PATH)
                    logger.info("TF-IDF vectorizer loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load TF-IDF vectorizer: {type(e).__name__}: {str(e)}")
                    raise
                
                try:
                    self.model = joblib.load(MODEL_PATH)
                    logger.info("Model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load model: {type(e).__name__}: {str(e)}")
                    raise
                
                self.is_trained = True
                logger.info("Pre-trained model loaded successfully")
                logger.info(f"Model type: {type(self.model)}")
                logger.info(f"Vectorizer type: {type(self.tfidf)}")
                return True
            
            # Try alternative path - check if running in HuggingFace Space
            # HF Spaces might mount the repo at a different location
            alt_model_path = Path("/home/user/app/models/sentiment_model.joblib")
            alt_tfidf_path = Path("/home/user/app/models/tfidf_vectorizer.joblib")
            
            if alt_model_path.exists() and alt_tfidf_path.exists():
                logger.warning("Using alternative HF Space path")
                try:
                    self.tfidf = joblib.load(alt_tfidf_path)
                    self.model = joblib.load(alt_model_path)
                    self.is_trained = True
                    logger.info("Pre-trained model loaded from HF Space path")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load from HF Space path: {type(e).__name__}: {str(e)}")
            
            logger.warning(f"Pre-trained model files not found. Checked paths:")
            logger.warning(f"  - {MODEL_PATH} (exists: {MODEL_PATH.exists()})")
            logger.warning(f"  - {TFIDF_PATH} (exists: {TFIDF_PATH.exists()})")
            logger.warning(f"  - {alt_model_path} (exists: {alt_model_path.exists()})")
            logger.warning(f"  - {alt_tfidf_path} (exists: {alt_tfidf_path.exists()})")
            return False
        except Exception as e:
            logger.error(f"Error loading pre-trained model: {type(e).__name__}: {str(e)}", exc_info=True)
            return False
    
    def train(self, data_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Train the sentiment analysis model.
        
        Args:
            data_path (Optional[Path]): Path to training data CSV file
            
        Returns:
            Dict[str, Any]: Training metrics and evaluation results
        """
        if data_path is None:
            data_path = TRAIN_DATA_PATH
        
        logger.info(f"Starting model training with data from {data_path}")
        
        # Load data
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} reviews")
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
        
        # Select relevant columns
        df = df[['Text', 'Score']].dropna()
        
        # Create sentiment labels (Score 4,5 -> Positive (1), Score 1,2,3 -> Negative (0))
        df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
        
        # Preprocess text
        logger.info("Preprocessing text...")
        df['Cleaned_Text'] = df['Text'].apply(self.preprocessor.clean_text)
        
        # TF-IDF Vectorization
        logger.info("Vectorizing text using TF-IDF...")
        self.tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
        X = self.tfidf.fit_transform(df['Cleaned_Text']).toarray()
        y = df['Sentiment'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Train model
        logger.info("Training Logistic Regression model...")
        self.model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Model trained. Accuracy: {metrics['accuracy']:.4f}")
        
        self.is_trained = True
        return metrics
    
    def save_model(self) -> bool:
        """
        Save the trained model and TF-IDF vectorizer to disk.
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.is_trained or self.model is None or self.tfidf is None:
            logger.error("Model not trained yet")
            return False
        
        try:
            # Create models directory if it doesn't exist
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.tfidf, TFIDF_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
            logger.info(f"TF-IDF vectorizer saved to {TFIDF_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text for sentiment prediction
            
        Returns:
            Tuple[str, float]: Sentiment label and confidence score
        """
        if not self.is_trained or self.model is None or self.tfidf is None:
            logger.error(f"Model not ready. is_trained={self.is_trained}, model={self.model is not None}, tfidf={self.tfidf is not None}")
            raise ValueError("Model not trained or loaded. Load or train model first.")
        
        try:
            # Preprocess
            cleaned_text = self.preprocessor.clean_text(text)
            logger.debug(f"Original: {text[:50]}... -> Cleaned: {cleaned_text[:50]}...")
            
            # Vectorize
            X = self.tfidf.transform([cleaned_text]).toarray()
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            
            sentiment_label = SENTIMENT_LABELS.get(prediction, "Unknown")
            logger.debug(f"Prediction: {sentiment_label} (confidence: {confidence:.4f})")
            
            return sentiment_label, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of dicts with 'text', 'sentiment', and 'confidence'
        """
        results = []
        for text in texts:
            try:
                sentiment, confidence = self.predict(text)
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence
                })
            except Exception as e:
                logger.error(f"Error predicting sentiment for text: {str(e)}")
                results.append({
                    'text': text,
                    'sentiment': 'Error',
                    'confidence': 0.0
                })
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'Text') -> pd.DataFrame:
        """
        Predict sentiment for a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing text
            text_column (str): Name of the column containing text
            
        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        results = []
        for text in df[text_column]:
            try:
                sentiment, confidence = self.predict(text)
                results.append({
                    'sentiment': sentiment,
                    'confidence': confidence
                })
            except Exception as e:
                logger.error(f"Error predicting: {str(e)}")
                results.append({
                    'sentiment': 'Error',
                    'confidence': 0.0
                })
        
        results_df = pd.DataFrame(results)
        return pd.concat([df, results_df], axis=1)


# Global model instance
_sentiment_model = None


def get_model() -> SentimentModel:
    """
    Get or create global sentiment model instance.
    
    Returns:
        SentimentModel: The sentiment model instance
    """
    global _sentiment_model
    
    if _sentiment_model is None:
        _sentiment_model = SentimentModel()
        if not _sentiment_model.load_pretrained_model():
            logger.warning("Could not load pre-trained model. Model will need to be trained.")
    
    return _sentiment_model
