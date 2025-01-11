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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from SentenceTransformerEmbeddings import SentenceTransformerEmbeddings

class ModelManager:
    """Manages language models and embeddings initialization and operations."""
    def __init__(self, env_path: str, mode: str= 'rag'):
        """Initialize ModelManager with environment variables."""
        self.mode = mode.lower()
        self.load_environment_variables(env_path)
        self.llm_type = None

    
    def load_environment_variables(self, env_path: str) -> None:
        """Load environment variables from .env file."""
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
        """Retrieve Pinecone API key"""
        return self.pinecone_api_key


    def get_openai_api_key(self) -> str:
        """Retrieve the OpenAI API key"""
        return self.openai_api_key


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
            elif embedding_type == 'SENTENCE_TRANSFORMER':
                return SentenceTransformerEmbeddings(model_name=model_name)
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")
                
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            sys.exit(1)


    def determine_embedding_dimensions(self, embeddings: Any) -> int:
        """Determine embedding dimensions using sample text."""
        try:
            text = "This is a text document."
            embedding = embeddings.embed_documents([text])[0]
            return len(embedding)
        except Exception as e:
            print(f"Error determining embedding dimensions: {e}")
            sys.exit(1)


    def validate_and_load_models(self, config: Dict[str, str], select_llm: str, select_embed: str, resource_manager: Any):
        """Validate and load selected models."""
        try:
            # Normalize config keys
            normalized_config = {
                key.upper(): value.upper() 
                for key, value in config.items()
            }
            # Get model selections - now using strings directly 
            selected_llm = select_llm
            selected_embedding_model = select_embed
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
            return model, embeddings, dimensions, selected_llm, selected_embedding_model
            
        except KeyError as e:
            print(f"Model selection error: {e}")
            raise