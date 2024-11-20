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
        """Clean and preprocess text for NLP tasks.
        
        Args:
            text (str): Raw input text to be cleaned and preprocessed.
            
        Returns:
            str: Processed text with standardized formatting, no stopwords, and lemmatized words.
            
        Raises:
            Exception: If an error occurs during preprocessing.
        """
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
        """Format text by adding newlines after colons and wrapping text.
        
        Args:
            text (str): Text to be formatted.
            line_length (int): Maximum line length before wrapping (default: 100).

        Returns:
            str: Formatted text with newlines after colons and word-wrapped.
            
        Raises:
            Exception: If text formatting fails.
        """
        try:
            # Add newline after colons
            text = text.replace(':', ':\n')
            
            # Format text with line wrapping
            formatted_text = ''
            current_length = 0
            
            for word in text.split(' '):
                word_length = len(word)
                
                if current_length + word_length > line_length:
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
                    current_length = word_length - word.rfind('\n') - 1
                    
            return formatted_text
        except Exception as e:
            self.logger.error(f"Error formatting text: {e}")
            raise

# Example usage
# preprocessor = TextPreprocessor()
# formatted_text = preprocessor.format_text("Your text here", line_length=80)

    # Function to save model output details and experimental setup to a text file.
    # Appends information about the model, embedding model, index, question, and answer.
    def save_results_to_file(model, embedding_model, index_name, question, answer):
        """Append model results and setup details to a text file.
        
        Implementation:
            1. Define filename based on the embedding model used, ensuring clear organization.
            2. Open the file in append mode to add details without overwriting existing data.
            3. Write the following information to the file:
                - Experimental setup, including model and embedding model names, and index.
                - Question posed to the model.
                - Model's answer to the question.
            4. Separate entries with a newline for readability in the results file.
            
        Args:
            model (str): Name or identifier of the large language model used.
            embedding_model (str): Name of the embedding model utilized.
            index_name (str): Name of the index involved in the experiment.
            question (str): Question that was posed to the model.
            answer (str): Model's response or answer to the question.
        
        File Output:
            - Creates or appends to a text file named "results_{embedding_model}.txt".
            - Each entry in the file documents the experimental setup and model's response.
            
        Note:
            This function is useful for logging experimental details and outputs in NLP or machine learning pipelines.
            It enables easy tracking of model parameters and results for later analysis.
        """
        
        # Define filename based on embedding model for organized file structure
        filename = f"results_{embedding_model}.txt"
        
        # Open file in append mode and write experiment details
        with open(filename, "a") as file:
            file.write(f"Experimental Setup - Large Language Model: {model}\n")
            file.write(f"                     Embedding Model: {embedding_model}\n")
            file.write(f"                     Index: {index_name}\n")
            file.write(f"Question: {question}\n")
            file.write(f"Answer: {answer}\n")
            file.write("\n")  # Add newline to separate entries
