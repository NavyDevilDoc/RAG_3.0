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

    Example:
        >>> manager = ModelManager(".env")
        >>> llm = manager.initialize_llm("gpt-4")
    """

    def __init__(self, env_path: str):
        """
        Initialize ModelManager with environment variables.

        Args:
            env_path (str): Path to .env file

        Raises:
            SystemExit: If required environment variables are missing
        """        
        self.load_environment_variables(env_path)
        self.llm_choices = self.define_llm_choices()
        self.embedding_choices = self.define_embedding_choices()

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
        """
        Define available language model choices.

        Returns:
            Dict[str, Any]: Available LLM configurations

        Example:
            >>> choices = manager.define_llm_choices()
            >>> print(choices["gpt"])  # "gpt-4o"
        """
        return {
            "gpt": "gpt-4o",
            "ollama": [
                "llama3.1:8b-instruct-q5_K_M",
                "llama3.2:latest",
                "mistral-nemo:12b-instruct-2407-q5_K_M"
            ]
        }

    def define_embedding_choices(self) -> Dict[str, List[str]]:
        """
        Define available embedding model choices.

        Returns:
            Dict[str, List[str]]: Available embedding model configurations

        Example:
            >>> choices = manager.define_embedding_choices()
            >>> print(choices["openai"])  # ["text-embedding-ada-002"]
        """
        return {
            "ollama": [
                "nomic-embed-text",
                "mxbai-embed-large",
                "all-minilm",
                "snowflake-arctic-embed"
            ],
            "gpt": [
                "text-embedding-3-small",
                "text-embedding-3-large",
            ],
            "sentence_transformer": [
                "all-MiniLM-L6-v2",
                "all-MiniLM-L12-v2",
                "all-mpnet-base-v2",
                "all-distilbert-base-v2",
                "multi-qa-mpnet-base-dot-v1"
            ]
        }


    def validate_selection(self, selection: str, choices: List[str]) -> None:
        """
        Validate model selection against available choices.

        Args:
            selection (str): Selected model type
            valid_choices (List[str]): List of valid options

        Raises:
            ValueError: If selection is invalid

        Example:
            >>> manager.validate_selection("gpt", ["gpt", "ollama"])
        """
        if selection not in choices:
            raise ValueError(f"Invalid selection: {selection}. Available choices are: {choices}")


    def load_model(self, selected_llm_type: str, selected_llm: str, resource_manager) -> Any:
        """
        Load and configure language model.

        Args:
            model_type (str): Type of model (e.g., "gpt", "ollama")
            model_name (str): Specific model name
            resource_manager (Any): Resource manager instance

        Returns:
            Any: Configured language model

        Raises:
            RuntimeError: If model loading fails

        Example:
            >>> model = manager.load_model("gpt", "gpt-4", resource_mgr)
        """
        try:
            if selected_llm_type == "gpt":
                return ChatOpenAI(openai_api_key=self.openai_api_key)
            else:
                return ChatOllama(model=selected_llm, **resource_manager)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)


    def load_embeddings(self, selected_embedding_scheme: str, selected_embedding_model: str) -> Any:
        """
        Load and configure embedding model.

        Args:
            embedding_type (str): Type of embeddings
            model_name (str): Name of embedding model

        Returns:
            Any: Configured embedding model

        Raises:
            RuntimeError: If embedding loading fails

        Example:
            >>> embeddings = manager.load_embeddings("openai", "ada-002")
        """
        try:
            if selected_embedding_scheme == "gpt":
                return OpenAIEmbeddings(
                    openai_api_key=self.openai_api_key, 
                    model=selected_embedding_model
                )
            elif selected_embedding_scheme == "ollama":
                return OllamaEmbeddings(model=selected_embedding_model)
            else:
                return SentenceTransformerEmbeddings(selected_embedding_model)
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

        Example:
            >>> dims = manager.determine_embedding_dimensions(embeddings)
            >>> print(f"Embedding dimensions: {dims}")
        """
        try:
            text = "This is a text document."
            embedding = embeddings.embed_documents([text])[0]
            return len(embedding)
        except Exception as e:
            print(f"Error determining embedding dimensions: {e}")
            sys.exit(1)


    def validate_and_load_models(
        self, config: Dict[str, str], select_llm: int, select_embed: int, resource_manager: Any
    ) -> tuple:
        """
        Validate and load selected language and embedding models.

        Args:
            config (Dict[str, str]): Model configuration
            select_llm (int): LLM selection index
            select_embed (int): Embedding selection index
            resource_manager (Any): Resource manager instance

        Returns:
            tuple: (
                model: Language model instance,
                embeddings: Embedding model instance,
                dimensions: Embedding dimensions,
                selected_llm: Selected LLM name,
                selected_embedding_model: Selected embedding model name
            )

        Raises:
            ValueError: If selections are invalid
            RuntimeError: If model loading fails

        Example:
            >>> model, emb, dims, llm, emb_model = manager.validate_and_load_models(
            ...     config, 0, 0, resource_mgr
            ... )
        """
        self.validate_selection(config["selected_llm_type"], self.llm_choices.keys())
        self.validate_selection(config["selected_embedding_scheme"], self.embedding_choices.keys())

        selected_llm = (
            self.llm_choices[config["selected_llm_type"]]
            if config["selected_llm_type"] == "gpt"
            else self.llm_choices[config["selected_llm_type"]][select_llm]
        )
        selected_embedding_model = self.embedding_choices[config["selected_embedding_scheme"]][select_embed]

        model = self.load_model(config["selected_llm_type"], selected_llm, resource_manager)
        embeddings = self.load_embeddings(config["selected_embedding_scheme"], selected_embedding_model)
        dimensions = self.determine_embedding_dimensions(embeddings)

        return model, embeddings, dimensions, selected_llm, selected_embedding_model
