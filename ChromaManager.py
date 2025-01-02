import chromadb
from typing import Any, Optional, List
from chromadb.config import Settings
from storage_constants import StorageType, is_chroma_storage, get_storage_mode


class ChromaManager:
    def __init__(self, persist_directory: str, embedding_model: Any):
        """Initialize ChromaDB manager with persistence and embedding config."""
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.embedding_model = embedding_model


    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(collection_name)
        except ValueError:
            return self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_model
            )

    def setup_collection(self, 
                        collection_name: str, 
                        documents: Optional[List] = None,
                        storage_type: StorageType = StorageType.CHROMA_NEW) -> Any:
        """Set up ChromaDB collection based on storage type."""
        try:
            if not is_chroma_storage(storage_type):
                raise ValueError(f"Invalid storage type for ChromaManager: {storage_type}")

            mode = get_storage_mode(storage_type)
            
            if mode == 'create':
                # Delete if exists and create new
                if collection_name in self.client.list_collections():
                    self.client.delete_collection(collection_name)
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_model
                )
            else:
                # Get existing or create new for 'add' and 'existing' modes
                collection = self._get_or_create_collection(collection_name)
            
            if documents:
                # Get current count for ID generation
                existing_count = collection.count() if mode == 'add' else 0
                
                # Process and add documents
                ids = [str(i + existing_count) for i in range(len(documents))]
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                
                collection.add(
                    documents=texts,
                    ids=ids,
                    metadatas=metadatas
                )
            
            return collection
            
        except Exception as e:
            raise RuntimeError(f"ChromaDB operation failed: {str(e)}")