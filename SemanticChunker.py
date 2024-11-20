from transformers import AutoTokenizer
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChunker:
    def __init__(self, chunk_size=200, chunk_overlap=0, similarity_threshold=0.9, separator=" ", sentence_model=None):
        """Initialize the semantic chunker with configurable parameters and tokenizer setup."""

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.separator = separator
        self.sentence_model = sentence_model or SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        self.text_splitter = SpacyTextSplitter(
            chunk_size=self.chunk_size - self.chunk_overlap,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the specified tokenizer."""
        try:
            if self.uses_tiktoken:
                return len(self.tokenizer.encode(text))
            elif self.uses_basic_tokenizer:
                return len(text.split())
            else:
                return len(self.tokenizer.tokenize(text))
        except Exception as e:
            print(f"Error counting tokens in text '{text[:30]}...': {e}")
            return 0

    def _get_semantic_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text."""
        if not text.strip():
            return None
        try:
            return self.sentence_model.encode(text)
        except Exception as e:
            print(f"Error generating embedding for text '{text[:30]}...': {e}")
            return None

    def get_semantic_chunks(self, documents):
        """Process documents into semantically coherent chunks within size limits."""
        base_chunks = self.text_splitter.split_documents(documents)
        print(f"Initial number of base chunks: {len(base_chunks)}")

        chunk_embeddings = [
            self._get_semantic_embedding(doc.page_content) for doc in base_chunks
        ]

        # Filter out any None embeddings
        valid_chunks = [chunk for chunk, emb in zip(base_chunks, chunk_embeddings) if emb is not None]
        chunk_embeddings = [emb for emb in chunk_embeddings if emb is not None]

        if not chunk_embeddings:
            print("No valid embeddings were generated.")
            return []

        grouped_chunks, current_group = [], []
        for i, base_chunk in enumerate(valid_chunks):
            if not current_group:
                current_group.append(base_chunk)
                current_embedding = chunk_embeddings[i].reshape(1, -1)
                continue

            # Calculate similarity
            similarity = cosine_similarity(current_embedding, chunk_embeddings[i].reshape(1, -1))[0][0]
            combined_content = " ".join([doc.page_content for doc in current_group] + [base_chunk.page_content])

            if similarity >= self.similarity_threshold and len(combined_content) <= self.chunk_size:
                current_group.append(base_chunk)
            else:
                grouped_chunks.extend(self._finalize_chunk_group(current_group))
                current_group = [base_chunk]
                current_embedding = chunk_embeddings[i].reshape(1, -1)

        if current_group:
            grouped_chunks.extend(self._finalize_chunk_group(current_group))
        
        if not grouped_chunks:
            print("No valid chunks were generated.")
            return []

        return grouped_chunks


    def _finalize_chunk_group(self, group):
        """Process and return finalized chunks from a group."""
        processed_chunks = []
        content = " ".join([doc.page_content for doc in group])
        size_limited_chunks = self._enforce_size_immediately(content)

        for chunk in size_limited_chunks:
            processed_chunks.append(Document(page_content=chunk, metadata=group[0].metadata))

        return processed_chunks

    def _enforce_size_immediately(self, text):
        """Split text into chunks strictly adhering to size limits."""
        # Replicate your existing `_enforce_size_immediately` logic here
        return []
