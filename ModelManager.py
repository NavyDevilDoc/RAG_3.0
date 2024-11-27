# ModelManager.py

"""
ModelManager.py

A module for managing language model and embedding configurations.

Features:
- Environment variable management
- LLM initialization and configuration
- Embedding model setup
- API key handling
"""

from typing import List, Any, Dict
import os
import sys
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from SentenceTransformerEmbeddings import SentenceTransformerEmbeddings

class ModelManager:
    """
    Manages language models and embeddings initialization and operations.

    Features:
    1. Environment variable loading
    2. LLM configuration
    3. Embedding model selection
    4. API key management

    Attributes:
        openai_api_key (str): OpenAI API key
        pinecone_api_key (str): Pinecone API key
        llm_choices (Dict[str, Any]): Available language models
        embedding_choices (Dict[str, Any]): Available embedding models
    """

    def __init__(self, env_path: str, mode: str= 'rag'):
        """
        Initialize ModelManager with environment variables.

        Args:
            env_path (str): Path to .env file

        Raises:
            SystemExit: If required environment variables are missing
        """
        self.mode = mode.lower()
        self.load_environment_variables(env_path)
        self.llm_choices = self.define_llm_choices()
        self.embedding_choices = self.define_embedding_choices()
        self.llm_type = None


    def load_environment_variables(self, env_path: str) -> None:
        """
        Load environment variables from .env file.

        Args:
            env_path (str): Path to .env file

        Raises:
            SystemExit: If required variables are missing
        """
        print("Loading environment variables...")
        load_dotenv(env_path)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")

        if self.openai_api_key and self.pinecone_api_key:
            print("Environment variables loaded successfully.")
        else:
            print("Error: Missing environment variables. Check the .env file.")
            sys.exit(1)


    def get_pinecone_api_key(self) -> str:
        """
        Retrieve Pinecone API key.

        Returns:
            str: Pinecone API key
        """
        return self.pinecone_api_key


    def get_openai_api_key(self) -> str:
        """Retrieve the OpenAI API key."""
        return self.openai_api_key


    def define_llm_choices(self) -> Dict[str, Any]:
        return {
            "GPT": "gpt-4o",
            "OLLAMA": [
                "llama3.1:8b-instruct-q5_K_M",
                "llama3.2:latest",
                "mistral-nemo:12b-instruct-2407-q5_K_M"
            ]
        }


    def define_embedding_choices(self) -> Dict[str, List[str]]:
        return {
            "GPT": [
                "text-embedding-3-small",
                "text-embedding-3-large",
            ],
            "SENTENCE_TRANSFORMER": [
                "all-MiniLM-L6-v2",
                "all-MiniLM-L12-v2",
                "all-mpnet-base-v2",
                "all-distilbert-base-v2",
                "multi-qa-mpnet-base-dot-v1"
            ]
        }


    def validate_selection(self, selection: str, valid_choices: List[str]) -> None:
        """Validate model selection with case-insensitive matching."""
        selection_upper = selection.upper()
        valid_choices_upper = [choice.upper() for choice in valid_choices]
        if selection_upper not in valid_choices_upper:
            raise ValueError(
                f"Invalid selection: {selection}. "
                f"Available choices are: {valid_choices}"
            )


    def load_model(self, selected_llm_type: str, selected_llm: str, resource_manager: Dict) -> Any:
        """Load and configure language model."""
        try:
            # Store normalized LLM type for embedding validation
            self.llm_type = selected_llm_type.upper()
            
            # Case insensitive comparison
            if selected_llm_type.lower() == "gpt":
                return ChatOpenAI(openai_api_key=self.openai_api_key, 
                                  temperature = 0.2,
                                  streaming=True,)
            else:
                return ChatOllama(model=selected_llm, 
                                  **resource_manager,
                                  disable_streaming=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)


    def load_embeddings(self, embedding_type: str, model_name: str) -> Any:
        """Load and configure embedding model with type validation."""
        try:
            embedding_type = embedding_type
            
            # Only enforce GPT-GPT pairing in LLM mode
            if self.mode == 'llm' and self.llm_type == 'GPT' and embedding_type != 'GPT':
                raise ValueError("GPT models require GPT embeddings in LLM mode")
            
            if embedding_type == 'GPT':
                return OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=self.openai_api_key
                )
            #elif embedding_type == 'OLLAMA':
                #return OllamaEmbeddings(model=model_name)
            elif embedding_type == 'SENTENCE_TRANSFORMER':
                return SentenceTransformerEmbeddings(model_name=model_name)
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")
                
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            sys.exit(1)


    def determine_embedding_dimensions(self, embeddings: Any) -> int:
        """
        Determine embedding dimensions using sample text.

        Args:
            embeddings (Any): Embedding model instance

        Returns:
            int: Embedding dimensions

        Raises:
            SystemExit: If dimension detection fails
        """
        try:
            text = "This is a text document."
            embedding = embeddings.embed_documents([text])[0]
            return len(embedding)
        except Exception as e:
            print(f"Error determining embedding dimensions: {e}")
            sys.exit(1)


    def validate_and_load_models(self, config: Dict[str, str], select_llm: int, select_embed: int, resource_manager: Any):
        """Validate and load selected models."""
        try:
            # Normalize config keys
            normalized_config = {
                key.upper(): value.upper() 
                for key, value in config.items()
            }
            
            # Get model selections
            selected_llm = self.llm_choices[normalized_config["SELECTED_LLM_TYPE"]][select_llm]
            selected_embedding_model = self.embedding_choices[normalized_config["SELECTED_EMBEDDING_SCHEME"]][select_embed]
            
            # Load models
            model = self.load_model(
                normalized_config["SELECTED_LLM_TYPE"], 
                selected_llm, 
                resource_manager
            )
            embeddings = self.load_embeddings(
                normalized_config["SELECTED_EMBEDDING_SCHEME"], 
                selected_embedding_model
            )
            
            # Get dimensions
            dimensions = self.determine_embedding_dimensions(embeddings)
            
            # Return all 5 expected values
            return model, embeddings, dimensions, selected_llm, selected_embedding_model
            
        except KeyError as e:
            print(f"Model selection error: {e}")
            print(f"Available LLM types: {list(self.llm_choices.keys())}")
            print(f"Available embedding types: {list(self.embedding_choices.keys())}")
            raise