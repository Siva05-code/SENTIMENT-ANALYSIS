"""
Text preprocessing utilities for sentiment analysis
"""

import re
import string
from pathlib import Path
from typing import List


class TextPreprocessor:
    """
    A class for preprocessing text data for sentiment analysis.
    
    This class provides methods for cleaning and normalizing text data
    including lowercase conversion, punctuation removal, and stopword filtering.
    """
    
    def __init__(self, language: str = 'english', remove_stopwords: bool = True):
        """
        Initialize the TextPreprocessor.
        
        Args:
            language (str): Language for stopwords. Default is 'english'.
            remove_stopwords (bool): Whether to remove stopwords. Default is True.
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        
        if remove_stopwords:
            self.stop_words = self._load_stopwords(language)
        else:
            self.stop_words = set()
    
    def _load_stopwords(self, language: str = 'english') -> set:
        """
        Load stopwords from local corpora or use common English stopwords.
        
        Args:
            language (str): Language for stopwords
            
        Returns:
            set: Set of stopwords
        """
        # Try to load from local corpora folder
        corpora_path = Path(__file__).parent.parent / 'corpora' / 'stopwords' / language
        
        if corpora_path.exists():
            try:
                with open(corpora_path, 'r', encoding='utf-8') as f:
                    words = set(line.strip() for line in f if line.strip())
                    return words
            except Exception:
                pass
        
        # Fallback to common English stopwords
        common_stopwords = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
            'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
            'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t',
            'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
            'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
            'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here',
            'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
            'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it',
            'it\'s', 'its', 'itself', 'just', 'k', 'let\'s', 'me', 'might', 'more', 'most',
            'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
            'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
            'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should',
            'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their',
            'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they',
            'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to',
            'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
            'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
            'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s',
            'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re',
            'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
        }
        return common_stopwords
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords
        if self.remove_stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        return text.split()
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text (str): Text containing URLs
            
        Returns:
            str: Text without URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def remove_emails(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text (str): Text containing emails
            
        Returns:
            str: Text without emails
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters from text.
        
        Args:
            text (str): Text with special characters
            
        Returns:
            str: Text without special characters
        """
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand common English contractions.
        
        Args:
            text (str): Text with contractions
            
        Returns:
            str: Text with expanded contractions
        """
        contractions_dict = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "could've": "could have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
        }
        
        pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
        return pattern.sub(lambda x: contractions_dict[x.group()], text.lower())


def preprocess_text(text: str) -> str:
    """
    Quick preprocessing function for single text.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    processor = TextPreprocessor()
    text = processor.remove_urls(text)
    text = processor.remove_emails(text)
    text = processor.expand_contractions(text)
    text = processor.clean_text(text)
    return text
