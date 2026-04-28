"""
Main entry point for the Sentiment Analysis API
"""

import sys
from pathlib import Path

# Add parent directory to path so backend module can be found
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app import app

__all__ = ["app"]
