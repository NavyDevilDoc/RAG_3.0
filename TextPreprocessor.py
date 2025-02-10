# TextPreprocessor.py

import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from typing import List

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
            words = self.remove_stopwords(words)
            words = self.lemmatize_words(words)
            
            return ' '.join(words)
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            raise


    def format_text(self, text: str, line_length: int = 100) -> str:
        """Format text with enhanced paragraph and list handling."""
        # First, clean up any existing formatting
        text = self._normalize_whitespace(text)
        
        # Split into sections (Answer: and Insights:)
        sections = text.split("Insights:")
        
        formatted_sections = []
        for i, section in enumerate(sections):
            if i > 0:  # This is the Insights section
                formatted_sections.append("Insights:")
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
            
            formatted_paragraphs = []
            for paragraph in paragraphs:
                # Handle bullet points
                if paragraph.strip().startswith("*"):
                    items = [item.strip() for item in paragraph.split("*") if item.strip()]
                    formatted_items = []
                    for item in items:
                        wrapped = self._wrap_text(item, line_length - 2)  # Account for "* " prefix
                        indented = self._indent_wrapped_text(wrapped, "* ", line_length)
                        formatted_items.append(indented)
                    formatted_paragraphs.append("\n".join(formatted_items))
                else:
                    # Normal paragraph
                    wrapped = self._wrap_text(paragraph, line_length)
                    formatted_paragraphs.append(wrapped)
            
            formatted_sections.append("\n\n".join(formatted_paragraphs))
        
        return "\n\n".join(formatted_sections)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving intentional line breaks."""
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        # Remove spaces at the beginning of lines
        text = re.sub(r'\n +', '\n', text)
        # Normalize line endings
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        return text

    def _wrap_text(self, text: str, line_length: int) -> str:
        """Wrap text at word boundaries."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= line_length:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)

    def _indent_wrapped_text(self, text: str, prefix: str, line_length: int) -> str:
        """Indent wrapped text with proper alignment."""
        lines = text.split('\n')
        if not lines:
            return ""
        
        # First line gets the prefix
        result = [prefix + lines[0]]
        # Subsequent lines get space alignment
        indent = ' ' * len(prefix)
        result.extend(indent + line for line in lines[1:])
        
        return '\n'.join(result)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        return len(word_tokenize(text))

'''
sample_text = "This is a sample text to test the format_text function. It should wrap text correctly at the specified line length. If the problem is not solved, review the solution and try again."
text_preprocessor = TextPreprocessor()
formatted_text = text_preprocessor.format_text(sample_text, line_length=100)
print(formatted_text)
'''