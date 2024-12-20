import time
import sys
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_pinecone import PineconeVectorStore

PINECONE_NEW = 0
PINECONE_ADD = 1
PINECONE_EXISTING = 2
LOCAL_STORAGE = 3

class PineconeManager:
    def __init__(self, pc, api_key, dimensions, selected_embedding_model):
        """
        Initialize the PineconeManager with Pinecone client and configurations.
        """
        self.pc = pc
        self.api_key = api_key
        self.dimensions = dimensions
        self.selected_embedding_model = selected_embedding_model


    def setup_index(self, index_name, spec, timeout=300):
        """
        Create or reset a Pinecone index.
            Pinecone index object.
        """
        try:
            if index_name in self.pc.list_indexes().names():
                print(f"\nDeleting existing index: {index_name}")
                self.pc.delete_index(index_name)

            print(f"\nCreating new index: {index_name}")
            self.pc.create_index(index_name, dimension=self.dimensions, metric='cosine', spec=spec)

            start_time = time.time()
            while not self.pc.describe_index(index_name).status['ready']:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Index creation timed out after {timeout} seconds")
                time.sleep(1)

            print(f"Index '{index_name}' ready for use")
            index = self.pc.Index(index_name)
            index.describe_index_stats()
            return index
        except Exception as e:
            print(f"\nError setting up Pinecone index: {e}")
            sys.exit(1)


    def setup_datastore(self, data_storage, documents, embeddings, index_name):
        """
        Configure the datastore based on the specified storage method.
        """
        try:
            if data_storage == PINECONE_NEW:
                print(f"\nUploading documents to new Pinecone index '{index_name}'")
                datastore = PineconeVectorStore.from_documents(documents, embedding=embeddings, index_name=index_name)

            elif data_storage == PINECONE_ADD:
                print(f"\nAdding documents to existing Pinecone index '{index_name}'")
                datastore = PineconeVectorStore.from_existing_index(index_name, embeddings)
                datastore.add_documents(documents)

            elif data_storage == PINECONE_EXISTING:
                print(f"\nUsing existing Pinecone index '{index_name}'")
                datastore = PineconeVectorStore.from_existing_index(index_name, embeddings)

            elif data_storage == LOCAL_STORAGE:
                print("\nStoring documents locally in memory")
                datastore = DocArrayInMemorySearch.from_documents(documents, embeddings)

            else:
                raise ValueError(f"Invalid data_storage option: {data_storage}")
            print("\nDatastore successfully configured.")
            return datastore
        
        except Exception as e:
            print(f"Error setting up datastore: {e}")
            sys.exit(1)