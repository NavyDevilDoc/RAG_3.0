# CombinedProcessor.py

"""
CombinedProcessor.py
A module for processing and storing documents with embeddings in vector databases.

Classes:
    CombinedProcessor: Handles document processing and storage with various RAG components.
"""

from typing import Union, List, Any
from pathlib import Path
import sys
from enum import Enum
from ChunkingInitializer import ChunkingInitializer
from ChunkingMethod import ChunkingMethod
from ModelManager import ModelManager
from DatastoreInitializer import DatastoreInitializer, StorageType


class CombinedProcessor:

    """
    A class that combines document processing, chunking, and vector storage operations.

    This class handles the end-to-end process of:
    1. Document ingestion
    2. Text chunking
    3. Embedding generation
    4. Vector storage management

    Attributes:
        doc_name (str): Unique identifier for the document index
        model_manager (ModelManager): Handles API key management
        embedding_model (str): Name of the embedding model to use
        embeddings (Any): Instance of the embedding model
        dimensions (int): Dimension size of the embeddings
        chunking_method (ChunkingMethod): Method for splitting documents
        enable_preprocessing (bool): Flag for text preprocessing
        storage_type (StorageType): Type of vector storage to use
        model_name (str): Name of the language model for processing

    Example:
        >>> from ModelManager import ModelManager
        >>> model_mgr = ModelManager()
        >>> processor = CombinedProcessor(
        ...     doc_name="my_docs",
        ...     model_manager=model_mgr,
        ...     embedding_model="text-embedding-ada-002",
        ...     embeddings=embeddings_instance,
        ...     dimensions=1536
        ... )
        >>> processor.process_and_store("path/to/documents")
    """

    def __init__(self, 
                 doc_name: str,
                 model_manager: ModelManager,  # Changed parameter name to match usage
                 embedding_model: str,
                 embeddings: Any,
                 dimensions: int,
                 chunking_method: ChunkingMethod = ChunkingMethod.SEMANTIC,
                 enable_preprocessing: bool = False,
                 storage_type: StorageType = StorageType.PINECONE_NEW,
                 model_name: str = None):
        """
        Initialize document processor with configuration and RAG components.
        
        Args:
            doc_name: Name for the document index
            model_manager: ModelManager instance for API keys
            embedding_model: Name of embedding model
            embeddings: Embedding model instance
            dimensions: Embedding dimensions
            chunking_method: Method for document chunking
            enable_preprocessing: Whether to preprocess documents
            storage_type: Type of storage to use
            model_name: Name of language model
        """
        self.doc_name = doc_name
        self.model_manager = model_manager
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.dimensions = dimensions
        self.chunking_method = chunking_method
        self.enable_preprocessing = enable_preprocessing
        self.storage_type = storage_type
        self.model_name = model_name
        
    def process_and_store(self, source_path: Union[str, List[str]]) -> None:
        """
        Process documents and store them in the vector database.

        This method handles:
        1. Connecting to existing storage (if specified)
        2. Processing new documents
        3. Storing document chunks with embeddings

        Args:
            source_path (Union[str, List[str]]): Path(s) to document(s) to process.
                Can be a single string path or list of paths.

        Returns:
            Optional[Any]: The initialized datastore object if successful, None otherwise.

        Raises:
            SystemExit: If critical errors occur during processing or storage
            Exception: For document processing or storage errors

        Example:
            >>> processor = CombinedProcessor(...)
            >>> datastore = processor.process_and_store("documents/file.pdf")
        """
        
        # Skip document processing for existing Pinecone index
        if self.storage_type == StorageType.PINECONE_EXISTING:
            print(f"\nConnecting to existing Pinecone index '{self.doc_name}'...")
            try:
                datastore_manager = DatastoreInitializer(
                    doc_name=self.doc_name,
                    pinecone_api_key=self.model_manager.get_pinecone_api_key(),
                    dimensions=self.dimensions,
                    embedding_model=self.embedding_model
                )
                
                datastore = datastore_manager.setup_datastore(
                    storage_type=self.storage_type,
                    documents=None,
                    embeddings=self.embeddings
                )
                
                print(f"Successfully connected to existing index '{self.doc_name}'")
                return datastore
                
            except Exception as e:
                print(f"Error connecting to existing index: {e}")
                sys.exit(1)
        
        # Process documents for new or additional storage
        paths = [source_path] if isinstance(source_path, str) else source_path
        all_documents = []
        
        for path in paths:
            processor = ChunkingInitializer(
                source_path=path,
                chunking_method=self.chunking_method,
                enable_preprocessing=self.enable_preprocessing,
                model_name=self.model_name,
                embedding_model=self.embedding_model
            )
            
            try:
                documents = processor.process()
                all_documents.extend(documents)
                print(f"\nProcessed {Path(path).name}")
            except Exception as e:
                print(f"Error processing {Path(path).name}: {e}")
                continue
        
        if not all_documents:
            print("No documents were successfully processed")
            return
            
        # Initialize storage
        try:
            datastore_manager = DatastoreInitializer(
                doc_name=self.doc_name,
                pinecone_api_key=self.model_manager.get_pinecone_api_key(),
                dimensions=self.dimensions,
                embedding_model=self.embedding_model
            )
            
            datastore = datastore_manager.setup_datastore(
                storage_type=self.storage_type,
                documents=all_documents,
                embeddings=self.embeddings
            )
            
            print(f"\nSuccessfully stored {len(all_documents)} chunks in '{self.doc_name}'")
                
        except Exception as e:
            print(f"Error storing documents: {e}")
            sys.exit(1)

        return datastore