"""
FastAPI Backend for Sentiment Analyzer
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import tempfile
import shutil

from backend.config import (
    CORS_ORIGINS, API_TITLE, API_VERSION, API_DESCRIPTION, SENTIMENT_EMOJI
)
from backend.sentiment_model import get_model
from backend.validate import ModelValidator, validate_input
from backend.preprocess import TextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic Models for request/response validation
class TextInput(BaseModel):
    """Model for single text sentiment analysis request"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class BatchTextInput(BaseModel):
    """Model for batch text sentiment analysis request"""
    texts: List[str] = Field(..., min_length=1, description="List of texts to analyze")


class SentimentResponse(BaseModel):
    """Model for sentiment analysis response"""
    text: str
    sentiment: str
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    emoji: str


class BatchSentimentResponse(BaseModel):
    """Model for batch sentiment analysis response"""
    results: List[SentimentResponse]
    total: int
    processing_time: float


class HealthResponse(BaseModel):
    """Model for health check response"""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    version: str
    model_trained: bool


class ModelStatsResponse(BaseModel):
    """Model for model statistics response"""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_type: str
    vectorizer: str
    max_features: int
    is_trained: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Sentiment Analyzer API")
    model = get_model()
    if model.is_trained:
        logger.info("Model loaded and ready")
    else:
        logger.warning("Model not trained. Please train the model.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment Analyzer API")


# Create results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    model = get_model()
    return {
        "status": "healthy",
        "version": API_VERSION,
        "model_trained": model.is_trained
    }


@app.get("/api/v1/model/stats", response_model=ModelStatsResponse)
async def get_model_stats():
    """
    Get model statistics and configuration.
    """
    stats = ModelValidator.get_model_statistics()
    return stats


@app.post("/api/v1/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: TextInput):
    """
    Analyze sentiment for a single text.
    
    Args:
        request: TextInput containing the text to analyze
        
    Returns:
        SentimentResponse with sentiment, confidence, and emoji
    """
    # Validate input
    is_valid, error_msg = validate_input(request.text)
    if not is_valid:
        logger.warning(f"Invalid input: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        model = get_model()
        if not model.is_trained:
            logger.error("Model not trained - cannot process request")
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Please train the model first."
            )
        
        sentiment, confidence = model.predict(request.text)
        emoji = SENTIMENT_EMOJI.get(sentiment, "")
        
        return {
            "text": request.text,
            "sentiment": sentiment,
            "confidence": float(confidence),
            "emoji": emoji
        }
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error analyzing sentiment")


@app.post("/api/v1/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(request: BatchTextInput):
    """
    Analyze sentiment for multiple texts.
    
    Args:
        request: BatchTextInput containing list of texts
        
    Returns:
        BatchSentimentResponse with results and processing time
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 texts allowed per request"
        )
    
    try:
        model = get_model()
        if not model.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Model not trained"
            )
        
        import time
        start_time = time.time()
        
        results = []
        for text in request.texts:
            # Validate each text
            is_valid, error_msg = validate_input(text)
            if not is_valid:
                results.append({
                    "text": text,
                    "sentiment": "Error",
                    "confidence": 0.0,
                    "emoji": "❌"
                })
                continue
            
            try:
                sentiment, confidence = model.predict(text)
                emoji = SENTIMENT_EMOJI.get(sentiment, "")
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": float(confidence),
                    "emoji": emoji
                })
            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                results.append({
                    "text": text,
                    "sentiment": "Error",
                    "confidence": 0.0,
                    "emoji": "❌"
                })
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "total": len(results),
            "processing_time": float(processing_time)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch")


@app.post("/api/v1/analyze/csv")
async def analyze_csv_file(file: UploadFile = File(...)):
    """
    Analyze sentiment for texts in an uploaded CSV file.
    
    Expected CSV columns: ProductId, UserId, Text
    
    Returns:
        CSV file with sentiment predictions
    """
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    try:
        # Read uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            
            # Validate CSV structure
            is_valid, error_msg = ModelValidator.validate_csv_file(tmp.name)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Process file
            model = get_model()
            if not model.is_trained:
                raise HTTPException(
                    status_code=503,
                    detail="Model not trained"
                )
            
            df = pd.read_csv(tmp.name)
            results_df = model.predict_dataframe(df, text_column='Text')
            
            # Save results to persistent location
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_filename = f"results_{unique_id}.csv"
            output_path = os.path.join(RESULTS_DIR, output_filename)
            results_df.to_csv(output_path, index=False)
            
            # Return results as response
            return {
                "status": "success",
                "rows_processed": len(results_df),
                "filename": output_filename,
                "download_url": f"/api/v1/download/{output_filename}",
                "preview": results_df.head(10).to_dict('records')
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing file")
    finally:
        # Cleanup
        if file:
            try:
                await file.close()
            except:
                pass


@app.get("/api/v1/download/{filename}")
async def download_results(filename: str):
    """
    Download CSV results file.
    
    Args:
        filename: Name of the results file to download
        
    Returns:
        CSV file as download
    """
    try:
        # Validate filename to prevent directory traversal attacks
        if "../" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.join(RESULTS_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Return file as download
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/csv'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading file")


@app.post("/api/v1/train")
async def train_model():
    """
    Train the sentiment analysis model.
    
    Note: This endpoint is for development/admin use.
    In production, model training should be done separately.
    """
    try:
        logger.info("Starting model training...")
        model = get_model()
        metrics = model.train()
        
        # Save model
        if model.save_model():
            return {
                "status": "success",
                "message": "Model trained and saved successfully",
                "metrics": metrics
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to save model"
            )
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/preprocess")
async def preprocess_text_endpoint(text: str):
    """
    Preprocess text (for debugging/testing).
    
    Args:
        text: Text to preprocess
        
    Returns:
        Preprocessed text
    """
    is_valid, error_msg = validate_input(text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        preprocessor = TextPreprocessor()
        cleaned = preprocessor.clean_text(text)
        return {
            "original": text,
            "cleaned": cleaned,
            "removed_chars": len(text) - len(cleaned)
        }
    except Exception as e:
        logger.error(f"Error preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error preprocessing text")


@app.get("/api/v1/status")
async def get_status():
    """
    Get detailed API status.
    """
    model = get_model()
    return {
        "api": "Sentiment Analyzer",
        "version": API_VERSION,
        "status": "running",
        "model": {
            "trained": model.is_trained,
            "type": "Logistic Regression",
            "vectorizer": "TF-IDF"
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /api/v1/analyze",
            "batch_analyze": "POST /api/v1/analyze/batch",
            "csv_analyze": "POST /api/v1/analyze/csv",
            "train": "POST /api/v1/train",
            "model_stats": "GET /api/v1/model/stats",
            "status": "GET /api/v1/status"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
