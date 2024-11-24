# ChunkingInitializer.py 

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
from ChunkingManager import ChunkingMethod, process_document

class ChunkingInitializer:
    """
    Orchestrates document processing workflow including OCR, preprocessing, and chunking.

    This class manages the end-to-end process of:
    1. Loading documents with OCR capabilities
    2. Optional text preprocessing
    3. Document chunking using various strategies

    Attributes:
        source_path (str): Path to source document(s)
        chunking_method (ChunkingMethod): Strategy for document chunking
        enable_preprocessing (bool): Whether to preprocess text
        chunk_size (int): Size of document chunks
        chunk_overlap (int): Overlap between chunks
        similarity_threshold (float): Threshold for semantic similarity
        model_name (Optional[str]): Name of language model
        embedding_model (Any): Model for semantic embeddings

    Example:
        >>> initializer = ChunkingInitializer(
        ...     source_path="docs/sample.pdf",
        ...     chunking_method=ChunkingMethod.SEMANTIC,
        ...     enable_preprocessing=True
        ... )
        >>> chunks = initializer.process()
    """
    
    def __init__(self, 
                 source_path: str,
                 chunking_method: ChunkingMethod = ChunkingMethod.PAGE,
                 enable_preprocessing: bool = False,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 similarity_threshold: float = 0.85,
                 model_name: Optional[str] = None,
                 embedding_model: Optional[Any] = None):
        """
        Initialize chunking processor with configuration parameters.

        Args:
            source_path (str): Path to document(s) to process
            chunking_method (ChunkingMethod): Method for splitting documents
            enable_preprocessing (bool): Whether to clean text
            chunk_size (int): Target size of chunks
            chunk_overlap (int): Overlap between chunks
            similarity_threshold (float): Semantic similarity threshold
            model_name (Optional[str]): Name of language model
            embedding_model (Optional[Any]): Model for embeddings

        Raises:
            ValueError: If parameters are invalid
        """
        self.source_path = source_path
        self.chunking_method = chunking_method
        self.enable_preprocessing = enable_preprocessing
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.embedding_model = embedding_model or self._setup_default_embedding()
        
    def _setup_default_embedding(self) -> any:
        """
        Setup default embedding model based on chunking method.

        Returns:
            Any: Configured embedding model

        Raises:
            RuntimeError: If model initialization fails
        """
        if self.chunking_method == ChunkingMethod.SEMANTIC:
            return SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        return None

    def _load_documents(self) -> List[Document]:
        """
        Load documents with OCR enhancement.

        Returns:
            List[Document]: List of loaded documents

        Raises:
            Exception: If document loading fails

        Example:
            >>> docs = initializer._load_documents()
            >>> print(f"Loaded {len(docs)} documents")
        """
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
        """
        Apply preprocessing to documents if enabled.

        Performs text cleaning and normalization on document content when preprocessing
        is enabled. Maintains original document metadata while adding preprocessing status.

        Args:
            documents (List[Document]): List of documents to preprocess

        Returns:
            List[Document]: Processed documents with updated metadata

        Raises:
            Exception: If preprocessing fails

        Example:
            >>> docs = initializer._load_documents()
            >>> processed = initializer._preprocess_documents(docs)
            >>> print(processed[0].metadata['preprocessing'])  # 'applied' or 'skipped'
        """
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
        """
        Execute the complete document processing pipeline.

        Performs the following steps:
        1. Load documents with OCR support
        2. Apply preprocessing if enabled
        3. Split documents into chunks using specified method
        4. Apply semantic processing if selected

        Returns:
            List[Document]: Processed and chunked documents

        Raises:
            Exception: If any processing step fails

        Example:
            >>> initializer = ChunkingInitializer("docs/sample.pdf")
            >>> chunks = initializer.process()
            >>> print(f"Generated {len(chunks)} chunks")
        """
        try:
            # Load and preprocess documents
            raw_documents = self._load_documents()
            processed_documents = self._preprocess_documents(raw_documents)
            
            # Process documents using specified chunking method
            documents = process_document(
                source_path=self.source_path,
                method=self.chunking_method,
                enable_preprocessing=self.enable_preprocessing,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                similarity_threshold=self.similarity_threshold,
                model_name=self.model_name,
                embedding_model=self.embedding_model
            )
            
            print(f"Processed {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            print(f"Error in document processing pipeline: {e}")
            raise