# DocumentEnhancement.py

"""
DocumentEnhancement.py

A module for document loading, preprocessing and enhancement.

Features:
- OCR-enhanced document loading
- Text preprocessing
- Document chunking
- Embedding integration
"""

from DocumentProcessor import DocumentProcessor
from langchain_core.documents import Document
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor
from ChunkingMethod import ChunkingMethod
from typing import List, Optional, Any


class DocumentEnhancement:
    """
    Orchestrates document loading, preprocessing, and chunking.

    Features:
    1. OCR-enabled document loading
    2. Text preprocessing options
    3. Flexible chunking strategies
    4. Embedding generation

    Attributes:
        embedding_model (Any): Model for generating embeddings
        processor (DocumentProcessor): Document processing handler

    Example:
        >>> enhancer = DocumentEnhancement(embedding_model=embeddings)
        >>> docs = enhancer.process_documents(
        ...     "docs/paper.pdf",
        ...     chunking_method="SEMANTIC"
        ... )
    """

    def __init__(self, embedding_model: Any):
        """
        Initialize DocumentEnhancement with embedding model.

        Args:
            embedding_model (Any): Model for generating embeddings

        Raises:
            ValueError: If embedding model is None
        """
        if embedding_model is None:
            raise ValueError("Embedding model must be provided.")
        self.embedding_model = embedding_model
        self.processor = DocumentProcessor(embedding_model=embedding_model)

    def load_and_preprocess_documents(
        self, source_path: str, enable_preprocessing: bool = False
    ) -> List[Document]:
        """
        Load and optionally preprocess documents from source.

        Args:
            source_path (str): Path to source document
            enable_preprocessing (bool): Whether to apply preprocessing

        Returns:
            List[Document]: Loaded and processed documents

        Raises:
            FileNotFoundError: If source file not found
            RuntimeError: If processing fails

        Example:
            >>> docs = enhancer.load_and_preprocess_documents(
            ...     "paper.pdf",
            ...     enable_preprocessing=True
            ... )
        """
        # Load document using OCR loader
        print("Loading documents with OCR enhancement...")
        ocr_loader = OCREnhancedPDFLoader(source_path)
        documents = ocr_loader.load()
        print(f"Loaded {len(documents)} pages.")

        # Apply preprocessing if enabled
        if enable_preprocessing:
            print("Preprocessing documents...")
            preprocess_text = TextPreprocessor().preprocess
            documents = [
                Document(
                    page_content=preprocess_text(doc.page_content),
                    metadata=doc.metadata,
                )
                for doc in documents
            ]
        else:
            print("Skipping preprocessing.")

        return documents

    def process_documents(
        self,
        source_path: str,
        chunking_method: str,
        enable_preprocessing: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85,
        model_name: Optional[str] = None,
    ) -> List[Document]:
        """
        Process documents with specified chunking method.

        Args:
            source_path (str): Path to source document
            chunking_method (str): Method for chunking ("SEMANTIC", "PAGE")
            enable_preprocessing (bool): Whether to preprocess text
            chunk_size (int): Size of chunks
            chunk_overlap (int): Overlap between chunks

        Returns:
            List[Document]: Processed document chunks

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If processing fails

        Example:
            >>> chunks = enhancer.process_documents(
            ...     "paper.pdf",
            ...     chunking_method="SEMANTIC",
            ...     chunk_size=500
            ... )
        """
        # Load and preprocess documents
        documents = self.load_and_preprocess_documents(
            source_path=source_path, enable_preprocessing=enable_preprocessing
        )

        # Delegate chunking to DocumentProcessor
        if chunking_method.upper() == "SEMANTIC":
            return self.processor.process_document(
                source_path,
                ChunkingMethod.SEMANTIC,
                enable_preprocessing,
                chunk_size,
                chunk_overlap,
                similarity_threshold,
            )
        elif chunking_method.upper() == "PAGE":
            return self.processor.process_document(
                source_path,
                ChunkingMethod.PAGE,
                enable_preprocessing,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unsupported chunking method: {chunking_method}")
