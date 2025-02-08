"""
ChunkingInitializer.py
A module for initializing and managing document chunking workflows.

This module handles the complete document processing pipeline including:
- Document loading with OCR support
- Text preprocessing
- Document chunking with various strategies
"""

from typing import List, Optional, Any
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor
from ChunkingManager import process_document
from storage_constants import ChunkingMethod

class ChunkingInitializer:
    """Orchestrates document processing workflow including OCR, preprocessing, and chunking."""
    def __init__(self, 
                 source_path: str,
                 chunking_method: ChunkingMethod = ChunkingMethod.PAGE,
                 enable_preprocessing: bool = True,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 similarity_threshold: float = 0.85,
                 model_name: Optional[str] = None,
                 embedding_model: Optional[Any] = None,
                 ):
        """Initialize chunking processor with configuration parameters."""
        self.source_path = source_path
        self.chunking_method = chunking_method
        self.enable_preprocessing = enable_preprocessing
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.embedding_model = embedding_model or self._setup_default_embedding()
        

    def _setup_default_embedding(self) -> any:
        """Setup default embedding model based on chunking method."""
        if self.chunking_method == ChunkingMethod.SEMANTIC:
            return SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        return None


    def _load_documents(self) -> List[Document]:
        """Load documents with OCR enhancement."""
        try:
            print("Loading documents with OCR enhancement...")
            loader = OCREnhancedPDFLoader(self.source_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents with OCR enhancement")
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise


    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Apply preprocessing to documents if enabled."""
        try:
            if self.enable_preprocessing:
                print("Preprocessing documents...")
                preprocessor = TextPreprocessor()
                return [
                    Document(
                        page_content=preprocessor.preprocess(doc.page_content),
                        metadata={**doc.metadata, "preprocessing": "applied"}
                    ) for doc in documents
                ]
            else:
                print("Skipping preprocessing...")
                return [
                    Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "preprocessing": "skipped"}
                    ) for doc in documents
                ]
        except Exception as e:
            print(f"Error in document preprocessing: {e}")
            raise


    def process(self) -> List[Document]:
        """Execute the complete document processing pipeline."""
        try:
            # Process documents using specified chunking method
            documents = process_document(
                source_path=self.source_path,
                method=self.chunking_method,
                enable_preprocessing=self.enable_preprocessing,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                similarity_threshold=self.similarity_threshold,
                model_name=self.model_name,
                embedding_model=self.embedding_model,
            )
            print(f"Processed {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            print(f"Error in document processing pipeline: {e}")
            raise