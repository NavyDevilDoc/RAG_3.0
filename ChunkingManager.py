# ChunkingManager provides document processing functionality for splitting and analyzing text content either by semantic similarity or page boundaries.

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from SemanticChunker import SemanticChunker
from PageChunker import PageChunker
from OCREnhancedPDFLoader import OCREnhancedPDFLoader
from TextPreprocessor import TextPreprocessor
from ChunkingMethod import ChunkingMethod
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import spacy
import numpy as np
from transformers import AutoTokenizer


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


def _page_chunking(source_path: str, preprocess: bool, model_name: str, embedding_model) -> List[Document]:

    print("Processing document by pages...")
    # Pass the pre-initialized embedding model to PageChunker
    page_chunker = PageChunker(model_name=model_name, embedding_model=embedding_model)
    documents = page_chunker.process_document(source_path, preprocess=preprocess)
    
    # Report token counts per page
    print(f"Processed {len(documents)} pages")
    for doc in documents:
        print(f"Page {doc.metadata['page']}: {doc.metadata['token_count']} tokens")
    
    return documents

# SemanticChunker segments documents into chunks that balance semantic coherence with size constraints.
# Processes documents for semantic search and analysis tasks, combining similarity-based merging with size enforcement.

class SemanticChunker:
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

        processed_chunks = []
        content = " ".join([doc.page_content for doc in group])
        size_limited_chunks = self._enforce_size_immediately(content)
        
        for chunk in size_limited_chunks:
            processed_chunks.append(Document(page_content=chunk, metadata=group[0].metadata))
        
        return processed_chunks



# PageChunker provides page-level document processing with token counting, embedding generation,
# and detailed page analysis. Skips blank pages, optionally preprocesses text, and maintains document structure.

# Load language model for semantic analysis using spaCy.
# This model performs core NLP tasks such as tokenization, lemmatization, and entity recognition.
nlp = spacy.load("en_core_web_sm")

class PageChunker:
    BLANK_THRESHOLD = 10  # Minimum character count to consider a page non-blank

    def __init__(self, model_name=None, embedding_model=None):

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
        cleaned_text = text.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        return len(cleaned_text) < self.BLANK_THRESHOLD

    def _count_tokens(self, text: str) -> int:
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

        if not text.strip():
            return None

        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            print(f"Error generating embedding for text '{text[:30]}...': {e}")
            return None
        
        
    def _analyze_page(self, text: str) -> dict:

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

        if self._is_blank_page(content):
            self.page_stats.append(f"Page {page_number} is blank.")
            return None

        # Optionally preprocess the text
        if preprocess:
            text_preprocessor = TextPreprocessor()
            content = text_preprocessor.preprocess(content)
        
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