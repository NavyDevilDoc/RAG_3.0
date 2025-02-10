from typing import List, Optional
import spacy
import numpy as np
from langchain_core.documents import Document
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor
from transformers import AutoTokenizer
import tiktoken

nlp = spacy.load("en_core_web_sm")

class ParagraphChunker:
    """Handles document chunking at the paragraph level with token counting."""
    
    PARAGRAPH_MIN_LENGTH = 50  # Minimum characters for a valid paragraph
    TOKEN_THRESHOLD = 30  # Minimum tokens for a valid paragraph
    
    def __init__(self, model_name=None, embedding_model=None):
        """
        Initialize page chunker with specified models."""
        self.model_name = model_name
        self.uses_tiktoken = False  # Default to False
        self.uses_basic_tokenizer = False  # Flag for Ollama models

        # List of models supported by tiktoken
        tiktoken_supported_models = [
            "gpt-3.5-turbo", "gpt-4", "text-davinci-003", "gpt-4o", "text-embedding-ada-002", "text-embedding-3-large"
        ]
        try:
            if model_name in tiktoken_supported_models:
                # Use tiktoken if the model is supported
                self.tokenizer = tiktoken.encoding_for_model(model_name)
                self.uses_tiktoken = True
            elif "mistral" in model_name or "llama" in model_name or "granite" in model_name:
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
        
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using NLP and multiple delimiter patterns."""
        # Pre-clean the text
        text = text.replace('\r', '\n')
        
        # Common paragraph delimiters
        delimiters = [
            '\n\n',           # Double line breaks
            '\n    ',         # Indented new lines
            '\n\t',          # Tab-indented new lines
            '\n•',           # Bullet points
            '\n-',           # Dashed lists
            '\n\d+\.',       # Numbered lists
        ]
        
        # Initial split using spaCy for sentence boundaries
        doc = nlp(text)
        potential_paragraphs = []
        current_paragraph = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Check if sentence starts a new paragraph
            starts_new = any(sent_text.startswith(d.strip()) for d in delimiters)
            if starts_new and current_paragraph:
                potential_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sent_text]
            else:
                current_paragraph.append(sent_text)
        
        # Add the last paragraph
        if current_paragraph:
            potential_paragraphs.append(' '.join(current_paragraph))
        
        # Filter and clean paragraphs
        cleaned_paragraphs = []
        for para in potential_paragraphs:
            clean_para = ' '.join(para.split())
            if len(clean_para) >= self.PARAGRAPH_MIN_LENGTH:
                cleaned_paragraphs.append(clean_para)
        
        return cleaned_paragraphs


    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the specified tokenizer."""
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


    def _process_single_paragraph(self, content: str, page_number: int, 
                                    para_number: int, preprocess: bool) -> Optional[Document]:
            """Process a single paragraph with analysis and metadata."""
            # First check character length
            if len(content.strip()) < self.PARAGRAPH_MIN_LENGTH:
                self.page_stats.append(f"Paragraph {para_number} on page {page_number} is too short.")
                return None
                
            if preprocess:
                content = TextPreprocessor().preprocess(content)
                
            stats = self._analyze_page(content)
            
            # Check token threshold
            if stats["token_count"] < self.TOKEN_THRESHOLD:
                self.page_stats.append(
                    f"Paragraph {para_number} on page {page_number} dropped: "
                    f"only {stats['token_count']} tokens"
                )
                return None
                
            metadata = {
                "page": page_number,
                "paragraph": para_number,
                "char_count": stats["char_count"],
                "token_count": stats["token_count"],
                "sentence_count": stats["sentence_count"],
                "word_count": stats["word_count"],
                "has_ocr": str(stats["has_ocr"])
            }
            
            return Document(page_content=content, metadata=metadata)
    

    def paragraph_process_document(self, file_path: str, preprocess: bool = False) -> List[Document]:
        """Process PDF document paragraph by paragraph with analysis."""
        try:
            loader = OCREnhancedPDFLoader(file_path)
            raw_pages = loader.load()
            processed_paragraphs = []
            
            for page_idx, page in enumerate(raw_pages):
                paragraphs = self._split_into_paragraphs(page.page_content)
                
                for para_idx, paragraph in enumerate(paragraphs):
                    processed_para = self._process_single_paragraph(
                        paragraph, 
                        page_idx + 1, 
                        para_idx + 1, 
                        preprocess
                    )
                    if processed_para:
                        processed_paragraphs.append(processed_para)
                        
            if self.page_stats:
                print("\n".join(self.page_stats))
                
            return processed_paragraphs
            
        except Exception as e:
            print(f"Error in paragraph_process_document: {e}")
            raise