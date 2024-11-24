"""
SemanticChunker.py
A module for semantic-aware text chunking using embeddings and similarity metrics.

This module provides functionality to:
- Split text into semantically coherent chunks
- Merge similar chunks based on cosine similarity
- Maintain chunk size constraints
- Calculate semantic similarity between text segments
"""

from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    """
    Chunks text based on semantic similarity and size constraints.

    This class combines traditional size-based chunking with semantic analysis to:
    1. Split text into initial chunks
    2. Calculate semantic similarity between chunks
    3. Merge similar chunks while respecting size limits
    4. Preserve semantic coherence in final chunks

    Attributes:
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Characters overlapping between chunks
        similarity_threshold (float): Minimum similarity to merge chunks
        separator (str): Chunk separation character
        sentence_model (SentenceTransformer): Model for embeddings
        text_splitter (SpacyTextSplitter): Spacy-based text splitter

    Example:
        >>> chunker = SemanticChunker(
        ...     chunk_size=200,
        ...     similarity_threshold=0.85
        ... )
        >>> chunks = chunker.get_semantic_chunks(documents)
    """

    def __init__(self, chunk_size=200, chunk_overlap=0, similarity_threshold=0.9, separator=" ", sentence_model=None):
        """Initialize the semantic chunker with configurable parameters.

        Args:
            chunk_size (int): Maximum size of each chunk in characters.
            chunk_overlap (int): Number of overlapping characters between chunks.
            similarity_threshold (float): Minimum cosine similarity score (0-1) to combine chunks.
            separator (str): Character used to separate chunks.
            sentence_model: Optional embedding model for similarity calculations.
        
        Raises:
            ValueError: If any initialization parameter is invalid.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if not (0 <= similarity_threshold <= 1):
            raise ValueError("similarity_threshold must be between 0 and 1.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.separator = separator
        self.sentence_model = sentence_model or SentenceTransformer('multi-qa-mpnet-base-dot-v1')  # Default model for flexibility
        self.text_splitter = SpacyTextSplitter(
            chunk_size=self.chunk_size - self.chunk_overlap,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator 
        )

    def _enforce_size_immediately(self, text):
        """
        Split text into chunks while strictly enforcing size limits.

        Args:
            text (str): Input text to chunk

        Returns:
            List[str]: List of text chunks

        Raises:
            ValueError: If input text is empty or invalid

        Example:
            >>> text = "Long document text here..."
            >>> chunks = chunker._enforce_size_immediately(text)
        """
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
        """
        Process documents into semantically coherent chunks.

        Workflow:
        1. Split documents into base chunks
        2. Generate embeddings for chunks
        3. Group similar chunks while respecting size limits
        4. Finalize and return processed chunks

        Args:
            documents (List[Document]): Input documents to process

        Returns:
            List[Document]: Semantically grouped document chunks

        Raises:
            ValueError: If documents list is empty
            RuntimeError: If embedding generation fails

        Example:
            >>> docs = [Document(page_content="Long text here...")]
            >>> chunks = chunker.get_semantic_chunks(docs)
            >>> print(f"Generated {len(chunks)} semantic chunks")
        """
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

            # Step 3: Calculate similarity and combine if appropriate
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
        """
        Process a group of related chunks into final documents.

        Args:
            group (List[Document]): Group of semantically related documents

        Returns:
            List[Document]: Processed and size-constrained documents

        Raises:
            ValueError: If group is empty
            RuntimeError: If chunk processing fails

        Example:
            >>> chunks = chunker._finalize_chunk_group(document_group)
            >>> print(f"Finalized {len(chunks)} chunks")
        """
        processed_chunks = []
        content = " ".join([doc.page_content for doc in group])
        size_limited_chunks = self._enforce_size_immediately(content)
        
        for chunk in size_limited_chunks:
            processed_chunks.append(Document(page_content=chunk, metadata=group[0].metadata))
        
        return processed_chunks