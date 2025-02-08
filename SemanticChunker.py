"""
SemanticChunker.py
A module for semantic-aware text chunking using embeddings and similarity metrics.

This module provides functionality to:
- Split text into semantically coherent chunks
- Merge similar chunks based on cosine similarity
- Maintain chunk size constraints
- Calculate semantic similarity between text segments
"""

from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor

class SemanticChunker:
    """Chunks text based on semantic similarity and size constraints"""
    def __init__(self, chunk_size=200, chunk_overlap=0, similarity_threshold=0.9, separator=" ", sentence_model=None):
        """Initialize the semantic chunker with configurable parameters"""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not (0 <= similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be between 0 and 1.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.separator = separator
        self.sentence_model = sentence_model or SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        self.text_splitter = SpacyTextSplitter(
            chunk_size=self.chunk_size - self.chunk_overlap,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator 
        )


    def _enforce_size_immediately(self, text):
        """Split text into chunks while strictly enforcing size limits"""
        if not text.strip():
            raise ValueError("Input 'text' cannot be empty or whitespace.")
        chunks, current_chunk = [], []
        words = text.split()
        for word in words:
            # Check if adding word would exceed size limit (including spaces)
            if sum(len(w) for w in current_chunk) + len(word) + len(current_chunk) <= self.chunk_size:
                current_chunk.append(word)
            else:
                # Save current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks


    def get_semantic_chunks(self, documents):
        """Process documents into semantically coherent chunks"""
        # Initial document splitting
        base_chunks = self.text_splitter.split_documents(documents)
        # Generate embeddings for semantic comparison
        chunk_embeddings = self.sentence_model.encode([doc.page_content for doc in base_chunks])
        grouped_chunks, current_group = [], []

        for i, base_chunk in enumerate(base_chunks):
            if not current_group:
                current_group.append(base_chunk)
                current_embedding = chunk_embeddings[i].reshape(1, -1)
                continue
            # Calculate similarity and combine if appropriate
            similarity = cosine_similarity(current_embedding, chunk_embeddings[i].reshape(1, -1))[0][0]
            combined_content = " ".join([doc.page_content for doc in current_group] + [base_chunk.page_content])

            if similarity >= self.similarity_threshold and len(combined_content) <= self.chunk_size:
                current_group.append(base_chunk)
            else:
                # Process current group and start a new one
                grouped_chunks.extend(self._finalize_chunk_group(current_group))
                current_group = [base_chunk]
                current_embedding = chunk_embeddings[i].reshape(1, -1)
        # Finalize any remaining chunks
        if current_group:
            grouped_chunks.extend(self._finalize_chunk_group(current_group))
        return grouped_chunks


    def _finalize_chunk_group(self, group):
        """Process a group of related chunks into final documents."""
        processed_chunks = []
        content = " ".join([doc.page_content for doc in group])
        size_limited_chunks = self._enforce_size_immediately(content)
        for chunk in size_limited_chunks:
            processed_chunks.append(Document(page_content=chunk, metadata=group[0].metadata))
        return processed_chunks
    

    def semantic_process_document(self, source_path: str, enable_preprocessing: bool = False) -> List[Document]:
        """Process document using semantic chunking strategy."""
        print("Performing semantic chunking...")
        # Load and preprocess document text
        ocr_loader = OCREnhancedPDFLoader(source_path)
        text_preprocessor = TextPreprocessor()
        raw_documents = ocr_loader.load()
        processed_documents = [
            Document(
                page_content=text_preprocessor.preprocess(doc.page_content) if enable_preprocessing else doc.page_content,
                metadata=doc.metadata
            ) for doc in raw_documents
        ]
        
        # Perform semantic chunking
        documents = self.get_semantic_chunks(processed_documents)
        print(f"Number of semantic chunks: {len(documents)}")
        return documents