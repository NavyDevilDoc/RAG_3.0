"""
PageChunker.py

A module for page-level document chunking with token counting and preprocessing.

Features:
- Page-based document splitting
- Multiple tokenizer support (tiktoken, HuggingFace, basic)
- Token counting and validation
- Blank page detection
- OCR integration
"""

from transformers import AutoTokenizer
import tiktoken
from typing import List, Optional, Tuple
import numpy as np
import spacy
from langchain_core.documents import Document
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor

nlp = spacy.load("en_core_web_sm")

class PageChunker:
    """
    Handles document chunking at the page level with token counting.
    """

    BLANK_THRESHOLD = 10  # Minimum character count to consider a page non-blank

    def __init__(self, model_name=None, embedding_model=None):
        """Initialize page chunker with specified models."""
        self.model_name = model_name
        self.uses_tiktoken = False  # Default to False
        self.uses_basic_tokenizer = False  # Flag for Ollama models

        # List of models supported by tiktoken
        tiktoken_supported_models = [
            "gpt-3.5-turbo", "gpt-4", "text-davinci-003", "gpt-4o", "text-embedding-ada-002", "text-embedding-3-large, text-embedding-3-small"
        ]

        try:
            if model_name in tiktoken_supported_models:
                # Use tiktoken if the model is supported
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                self.uses_tiktoken = True
            elif "mistral" in model_name or "llama" in model_name:
                # Handle Ollama models with a basic tokenizer
                self.uses_basic_tokenizer = True
            else:
                # Fall back to transformers tokenizer for supported Hugging Face models
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.uses_tiktoken = False

            # Check if embedding model is provided
            if embedding_model is None:
                raise ValueError("Embedding model must be provided.")
            self.embedding_model = embedding_model

        except Exception as e:
            raise ValueError(f"Error initializing PageChunker with '{model_name}': {e}")
        self.page_stats = []


    def _is_blank_page(self, text: str) -> bool:
        """Check if page is blank or contains only whitespace/special characters."""
        cleaned_text = text.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        return len(cleaned_text) < self.BLANK_THRESHOLD


    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the specified tokenizer. """
        try:
            if self.uses_tiktoken:
                # Use tiktoken to count tokens
                return len(self.tokenizer.encode(text))
            elif self.uses_basic_tokenizer:
                # Basic token counting for Ollama models
                return len(text.split())
            else:
                # Use transformers tokenizer for supported Hugging Face models
                return len(self.tokenizer.tokenize(text))
        except Exception as e:
            print(f"Error counting tokens in text '{text[:30]}...': {e}")
            return 0


    def _get_page_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding vector for page text."""
        if not text.strip():
            return None
        
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            return None
        
        
    def _analyze_page(self, text: str) -> dict:
        """Perform detailed analysis of page content."""
        
        try:
            embedding = self._get_page_embedding(text)
            return {
                "char_count": len(text),
                "token_count": self._count_tokens(text),
                "sentence_count": len(list(nlp(text).sents)),
                "word_count": len(text.split()),
                "embedding_dim": len(embedding) if embedding is not None else 0,
                "has_ocr": bool(text.strip())
            }
        except Exception as e:
            print(f"Error analyzing page: {e}")
            return {
                "char_count": 0,
                "token_count": 0,
                "sentence_count": 0,
                "word_count": 0,
                "embedding_dim": 0,
                "has_ocr": False
            }

    def _process_single_page(self, content: str, page_number: int, preprocess: bool) -> Optional[Document]:
        """Process a single page with optional preprocessing and analysis.
        
        Args:
            content (str): Page content to process.
            page_number (int): Page number for metadata.
            preprocess (bool): Whether to preprocess the text.
            
        Returns:
            Optional[Document]: Processed Document object, or None if page is blank.
        """
        if self._is_blank_page(content):
            self.page_stats.append(f"Page {page_number} is blank.")
            return None

        # Optionally preprocess the text
        if preprocess:
            preprocess_text = TextPreprocessor().preprocess
            content = preprocess_text(content)
        
        # Analyze the page and generate metadata
        stats = self._analyze_page(content)
        metadata = {
            "page": page_number,
            "char_count": stats["char_count"],
            "token_count": stats["token_count"],
            "sentence_count": stats["sentence_count"],
            "word_count": stats["word_count"],
            "has_ocr": str(stats["has_ocr"]),
            "is_blank": "false"
        }
        
        return Document(page_content=content, metadata=metadata)

    def process_document(self, file_path: str, preprocess: bool = False) -> List[Document]:
        """Process PDF document page by page with analysis and optional preprocessing.
        
        Args:
            file_path (str): Path to PDF file.
            preprocess (bool): Whether to apply text preprocessing.
            
        Returns:
            List[Document]: List of processed pages as Document objects.
            
        Raises:
            Exception: If error occurs during processing.
        """
        try:
            loader = OCREnhancedPDFLoader(file_path)
            raw_pages = loader.load()
            processed_pages = []

            for idx, page in enumerate(raw_pages):
                processed_page = self._process_single_page(page.page_content, idx + 1, preprocess)
                if processed_page:
                    processed_pages.append(processed_page)

            # Output skipped pages for transparency
            if self.page_stats:
                print("\n".join(self.page_stats))

            return processed_pages
            
        except Exception as e:
            print(f"Error in process_document: {e}")
            raise
