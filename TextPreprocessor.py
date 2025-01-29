# TextPreprocessor.py

import re
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class TextPreprocessor:
    """A class to preprocess text for NLP tasks."""
    
    def __init__(self):
        """Initialize the text preprocessor with required NLTK resources."""
        try:
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            self.logger.error(f"Failed to initialize NLTK resources: {e}")
            raise

    def standardize_case(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def remove_punctuation(self, text):
        """Remove punctuation and non-word characters."""
        return re.sub(r'[^\w\s]', '', text)

    def normalize_whitespace(self, text):
        """Normalize whitespace by collapsing multiple spaces and trimming."""
        return re.sub(r'\s+', ' ', text).strip()

    def remove_stopwords(self, words):
        """Remove stopwords from list of words."""
        return [word for word in words if word not in self.stopwords]

    def lemmatize_words(self, words):
        """Lemmatize list of words."""
        return [self.lemmatizer.lemmatize(word) for word in words]

    def preprocess(self, text):
        """Clean and preprocess text for NLP tasks."""
        try:
            # Apply preprocessing steps
            text = self.standardize_case(text)
            text = self.remove_punctuation(text)
            text = self.normalize_whitespace(text)
            
            # Tokenize and process words
            words = text.split()
            #words = self.remove_stopwords(words)
            #words = self.lemmatize_words(words)
            
            return ' '.join(words)
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            raise

    def format_text(self, text: str, line_length: int = 100) -> str:
        """Format text by adding newlines after colons and wrapping text."""
        
        try:
            # Add newline after colons
            text = text.replace(':', ':\n')
            
            # Format text with line wrapping
            words = text.split(' ')
            formatted_text = ''
            current_length = 0
            
            for word in text.split(' '):
                word_length = len(word)
                
                if current_length + word_length + 1 > line_length:
                    formatted_text += '\n' + word
                    current_length = word_length
                else:
                    if formatted_text:
                        formatted_text += ' '
                        current_length += 1
                    formatted_text += word
                    current_length += word_length
                
                # Reset length after newlines
                if '\n' in word:
                    current_length = len(word.split('\n')[-1])
                    
            return formatted_text
        except Exception as e:
            self.logger.error(f"Error formatting text: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(word_tokenize(text))

'''
sample_text = "This is a sample text to test the format_text function. It should wrap text correctly at the specified line length. If the problem is not solved, review the solution and try again."
text_preprocessor = TextPreprocessor()
formatted_text = text_preprocessor.format_text(sample_text, line_length=100)
print(formatted_text)
'''