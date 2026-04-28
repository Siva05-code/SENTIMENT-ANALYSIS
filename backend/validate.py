"""
Model Validation and Evaluation Utilities
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Any, Tuple

from backend.config import TEST_DATA_PATH, OUTPUT_PATH
from backend.sentiment_model import get_model

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Utilities for validating and evaluating the sentiment analysis model.
    
    Provides methods for:
    - Performance metrics calculation
    - Confusion matrix generation
    - Cross-validation
    - Test set evaluation
    """
    
    @staticmethod
    def evaluate_on_test_set(test_data_path: Path = TEST_DATA_PATH) -> Dict[str, Any]:
        """
        Evaluate model performance on a test dataset.
        
        Args:
            test_data_path (Path): Path to test CSV file
            
        Returns:
            Dict[str, Any]: Evaluation metrics and results
        """
        try:
            df = pd.read_csv(test_data_path)
            logger.info(f"Loaded test dataset with {len(df)} samples")
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
        
        # Ensure required columns exist
        required_cols = ['Text', 'Score']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Test data must contain columns: {required_cols}")
        
        # Create labels
        df['True_Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
        
        # Get predictions
        model = get_model()
        predictions = []
        confidences = []
        
        for text in df['Text']:
            try:
                sentiment, confidence = model.predict(text)
                pred_label = 1 if sentiment == "Positive" else 0
                predictions.append(pred_label)
                confidences.append(confidence)
            except Exception as e:
                logger.error(f"Error predicting: {str(e)}")
                predictions.append(0)
                confidences.append(0.0)
        
        df['Predicted_Sentiment'] = predictions
        df['Confidence'] = confidences
        
        # Calculate metrics
        metrics = ModelValidator.calculate_metrics(
            df['True_Sentiment'].values,
            df['Predicted_Sentiment'].values,
            confidences
        )
        
        # Save results
        try:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUTPUT_PATH, index=False)
            logger.info(f"Test results saved to {OUTPUT_PATH}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
        
        metrics['total_samples'] = len(df)
        
        return metrics
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         confidences: list = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            confidences (list): Confidence scores (optional)
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Average confidence
        if confidences:
            metrics['avg_confidence'] = float(np.mean(confidences))
            metrics['min_confidence'] = float(np.min(confidences))
            metrics['max_confidence'] = float(np.max(confidences))
        
        return metrics
    
    @staticmethod
    def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            str: Classification report
        """
        target_names = ['Negative', 'Positive']
        return classification_report(y_true, y_pred, target_names=target_names)
    
    @staticmethod
    def validate_text_input(text: str) -> Tuple[bool, str]:
        """
        Validate input text for analysis.
        
        Args:
            text (str): Text to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check if text is empty
        if not text or not text.strip():
            return False, "Text cannot be empty"
        
        # Check if text is too short
        if len(text.strip()) < 3:
            return False, "Text must be at least 3 characters long"
        
        # Check if text is too long
        if len(text) > 10000:
            return False, "Text cannot exceed 10000 characters"
        
        return True, ""
    
    @staticmethod
    def validate_csv_file(file_path: Path) -> Tuple[bool, str]:
        """
        Validate CSV file structure.
        
        Args:
            file_path (Path): Path to CSV file
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_cols = ['ProductId', 'UserId', 'Text']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return False, f"Missing columns: {', '.join(missing_cols)}"
            
            # Check for empty DataFrame
            if df.empty:
                return False, "CSV file is empty"
            
            # Check for missing Text values
            if df['Text'].isna().any():
                return False, "Some Text values are missing"
            
            return True, ""
        
        except Exception as e:
            return False, f"Error reading CSV file: {str(e)}"
    
    @staticmethod
    def get_model_statistics() -> Dict[str, Any]:
        """
        Get statistics about the trained model.
        
        Returns:
            Dict[str, Any]: Model statistics
        """
        model = get_model()
        
        if not model.is_trained:
            return {'status': 'Model not trained'}
        
        stats = {
            'status': 'Ready',
            'model_type': 'Logistic Regression',
            'vectorizer': 'TF-IDF',
            'max_features': 5000,
            'is_trained': True,
        }
        
        return stats


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Quick validation function for input text.
    
    Args:
        text (str): Text to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    return ModelValidator.validate_text_input(text)
