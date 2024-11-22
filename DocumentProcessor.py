# DocumentProcessor.py

"""
DocumentProcessor.py

A module for processing documents with various chunking strategies and OCR support.

Features:
- Multiple chunking methods (semantic and page-based)
- OCR-enhanced document loading
- Text preprocessing options
- Embedding integration
"""

from typing import List, Any, Optional
from langchain_core.documents import Document
from ChunkingMethod import ChunkingMethod
from SemanticChunker import SemanticChunker
from PageChunker import PageChunker
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor

class DocumentProcessor:
    """
    Processes documents using specified chunking methods and parameters.

    Features:
    1. Multiple chunking strategies
    2. OCR support
    3. Text preprocessing
    4. Embedding generation

    Attributes:
        embedding_model (Any): Model for generating embeddings

    Example:
        >>> processor = DocumentProcessor(embedding_model=embeddings)
        >>> chunks = processor.process_document(
        ...     "doc.pdf",
        ...     ChunkingMethod.SEMANTIC
        ... )
    """

    def __init__(self, embedding_model: Any):
        """
        Initialize DocumentProcessor with an embedding model.

        Args:
            embedding_model (Any): Model for generating embeddings

        Raises:
            ValueError: If embedding model is None
        """
        if embedding_model is None:
            raise ValueError("Embedding model must be provided.")
        self.embedding_model = embedding_model

    def process_document(
        self,
        source_path: str,
        method: ChunkingMethod, 
        enable_preprocessing: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85,
        model_name: Optional[str] = None
    ) -> List[Document]:
        """
        Process document using specified chunking method.

        Args:
            source_path (str): Path to source document
            method (ChunkingMethod): Chunking strategy to use
            enable_preprocessing (bool): Whether to preprocess text
            chunk_size (int): Size of chunks
            chunk_overlap (int): Overlap between chunks
            similarity_threshold (float): Threshold for semantic similarity
            model_name (Optional[str]): Name of language model

        Returns:
            List[Document]: Processed document chunks

        Raises:
            ValueError: If method is invalid
            RuntimeError: If processing fails

        Example:
            >>> chunks = processor.process_document(
            ...     "paper.pdf",
            ...     ChunkingMethod.SEMANTIC,
            ...     chunk_size=500
            ... )
        """
        try:
            if method == ChunkingMethod.SEMANTIC:
                return self._semantic_chunking(
                    source_path,
                    enable_preprocessing,
                    chunk_size,
                    chunk_overlap,
                    similarity_threshold,
                )
            elif method == ChunkingMethod.PAGE:
                return self._page_chunking(source_path, enable_preprocessing, model_name)
            else:
                raise ValueError(f"Unsupported chunking method: {method}")

        except Exception as e:
            print(f"Error processing document with {method.name} chunking: {e}")
            raise RuntimeError("Document processing failed") from e

    def _semantic_chunking(
        self,
        source_path: str,
        enable_preprocessing: bool,
        chunk_size: int,
        chunk_overlap: int,
        similarity_threshold: float,
    ) -> List[Document]:
        """
        Process document using semantic chunking.

        Args:
            source_path (str): Path to source document
            enable_preprocessing (bool): Whether to preprocess text
            chunk_size (int): Size of chunks
            chunk_overlap (int): Overlap between chunks
            similarity_threshold (float): Semantic similarity threshold

        Returns:
            List[Document]: Semantically chunked documents

        Raises:
            RuntimeError: If semantic chunking fails
        """
        print("Performing semantic chunking...")
        semantic_chunker = SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold,
            separator=" ",
        )
        
        # Load and preprocess document text if needed
        ocr_loader = OCREnhancedPDFLoader(source_path)
        preprocess_text = TextPreprocessor().preprocess
        raw_documents = ocr_loader.load()
        print(f"Loaded {len(raw_documents)} pages")
        if raw_documents:
            print(f"Sample page content: {raw_documents[0].page_content[:200]}...")

        processed_documents = [
            Document(
                page_content=preprocess_text(doc.page_content) if enable_preprocessing else doc.page_content,
                metadata=doc.metadata
            ) for doc in raw_documents
        ]

        print(f"Number of processed documents: {len(processed_documents)}")
        if processed_documents:
            print(f"Sample processed content: {processed_documents[0].page_content[:200]}...")

        
        # Perform semantic chunking and return results
        documents = semantic_chunker.get_semantic_chunks(processed_documents)
        print(f"Number of semantic chunks: {len(documents)}")
        return documents

    def _page_chunking(self, source_path: str, preprocess: bool, model_name: str) -> List[Document]:
        """
        Process document using page-based chunking.

        Args:
            source_path (str): Path to source document
            enable_preprocessing (bool): Whether to preprocess text
            model_name (Optional[str]): Name of language model

        Returns:
            List[Document]: Page-chunked documents

        Raises:
            RuntimeError: If page chunking fails
        """
        # Pass the pre-initialized embedding model to PageChunker
        page_chunker = PageChunker(model_name=model_name, embedding_model=self.embedding_model)
        documents = page_chunker.process_document(source_path, preprocess=preprocess)
        
        # Report token counts per page
        print(f"Processed {len(documents)} pages")
        for doc in documents:
            print(f"Page {doc.metadata['page']}: {doc.metadata['token_count']} tokens")
        
        return documents