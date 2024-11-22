# ChunkingManager.py 

"""
ChunkingManager.py
A module for managing different document chunking strategies.

This module provides functionality to:
- Process documents using different chunking methods
- Support semantic and page-based chunking
- Handle preprocessing and OCR integration
"""

from typing import List, Any
from langchain_core.documents import Document
from SemanticChunker import SemanticChunker
from PageChunker import PageChunker
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor
from ChunkingMethod import ChunkingMethod


def process_document(
    source_path: str,
    method: ChunkingMethod, 
    enable_preprocessing: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    similarity_threshold: float = 0.85,
    max_tokens: int = 8000,
    model_name = None,
    embedding_model=None
) -> List[Document]:

    """
    Process documents using specified chunking method.

    Args:
        source_path (str): Path to source document
        method (ChunkingMethod): Chunking strategy to use
        enable_preprocessing (bool): Whether to preprocess text
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        similarity_threshold (float): Threshold for semantic similarity
        max_tokens (int): Maximum tokens per chunk
        model_name (Optional[str]): Name of language model
        embedding_model (Optional[Any]): Model for embeddings

    Returns:
        List[Document]: Processed document chunks

    Raises:
        ValueError: If embedding model is missing or method invalid
        RuntimeError: If document processing fails

    Example:
        >>> chunks = process_document(
        ...     "doc.pdf",
        ...     ChunkingMethod.SEMANTIC,
        ...     embedding_model=embeddings
        ... )
    """

    if embedding_model is None:
        raise ValueError("Embedding model must be provided.")

    try:
        if method == ChunkingMethod.SEMANTIC:
            return _semantic_chunking(
                source_path,
                enable_preprocessing,
                chunk_size,
                chunk_overlap,
                similarity_threshold,
            )
        elif method == ChunkingMethod.PAGE:
            return _page_chunking(source_path, enable_preprocessing, model_name, embedding_model)
        else:
            raise ValueError(f"Unsupported chunking method: {method}")

    except Exception as e:
        print(f"Error processing document with {method.name} chunking: {e}")
        raise RuntimeError("Document processing failed") from e


def _semantic_chunking(
    source_path: str,
    enable_preprocessing: bool,
    chunk_size: int,
    chunk_overlap: int,
    similarity_threshold: float,
) -> List[Document]:

    """
    Process document using semantic chunking strategy.

    Args:
        source_path (str): Path to source document
        enable_preprocessing (bool): Whether to preprocess text
        chunk_size (int): Size of chunks
        chunk_overlap (int): Overlap between chunks
        similarity_threshold (float): Semantic similarity threshold

    Returns:
        List[Document]: Semantically chunked documents

    Raises:
        Exception: If semantic chunking fails
    """

    print("Performing semantic chunking...")
    semantic_chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
        separator=" "
    )
    
    # Load and preprocess document text if needed
    ocr_loader = OCREnhancedPDFLoader(source_path)
    text_preprocessor = TextPreprocessor()
    raw_documents = ocr_loader.load()
    processed_documents = [
        Document(
            page_content=text_preprocessor.preprocess(doc.page_content) if enable_preprocessing else doc.page_content,
            metadata=doc.metadata
        ) for doc in raw_documents
    ]
    
    # Perform semantic chunking and return results
    documents = semantic_chunker.get_semantic_chunks(processed_documents)
    print(f"Number of semantic chunks: {len(documents)}")
    return documents


def _page_chunking(
        source_path: str, 
        preprocess: bool, 
        model_name: str, 
        embedding_model: Any
        ) -> List[Document]:

    """
    Process document using page-based chunking strategy.

    This method:
    1. Splits document by pages
    2. Optionally preprocesses text
    3. Calculates token counts per page
    4. Applies embeddings for retrieval

    Args:
        source_path (str): Path to source document
        preprocess (bool): Whether to preprocess text
        model_name (str): Name of language model
        embedding_model (Any): Model for generating embeddings

    Returns:
        List[Document]: List of page-chunked documents with metadata

    Raises:
        ValueError: If document or model parameters are invalid
        RuntimeError: If chunking process fails

    Example:
        >>> chunks = _page_chunking(
        ...     "doc.pdf",
        ...     preprocess=True,
        ...     model_name="gpt-3.5-turbo",
        ...     embedding_model=embeddings
        ... )
        >>> print(f"Processed {len(chunks)} pages")
    """

    print("Processing document by pages...")
    # Pass the pre-initialized embedding model to PageChunker
    page_chunker = PageChunker(model_name=model_name, embedding_model=embedding_model)
    documents = page_chunker.process_document(source_path, preprocess=preprocess)
    
    # Report token counts per page
    print(f"Processed {len(documents)} pages")
    for doc in documents:
        print(f"Page {doc.metadata['page']}: {doc.metadata['token_count']} tokens")
    
    return documents