"""
CombinedProcessor.py
A module for processing and storing documents with embeddings in vector databases.

Classes:
    CombinedProcessor: Handles document processing and storage with various RAG components.
"""

from typing import Union, List, Any
from pathlib import Path
import sys
from ChunkingInitializer import ChunkingInitializer
from storage_constants import ChunkingMethod
from ModelManager import ModelManager
from DatastoreInitializer import DatastoreInitializer
from storage_constants import StorageType

class CombinedProcessor:
    def __init__(self, 
                 doc_name: str,
                 model_manager: ModelManager,
                 embedding_model: str,
                 embeddings: Any,
                 dimensions: int,
                 chunking_method: ChunkingMethod = ChunkingMethod.SEMANTIC,
                 enable_preprocessing: bool = True,
                 storage_type: StorageType = StorageType.PINECONE_NEW,
                 model_name: str = None,
                 ):
        """Initialize document processor with configuration and RAG components."""
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
        """Process documents and store them in the vector database."""
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
        # Process documents for new or additional storage (PINECONE_NEW or PINECONE_ADD)
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