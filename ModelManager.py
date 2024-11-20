# ModelManager (Test)

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
    """Manages language models and embeddings initialization and operations."""

    def __init__(self, env_path: str):
        """Initialize ModelManager with environment variables and logging."""
        self.load_environment_variables(env_path)
        self.llm_choices = self.define_llm_choices()
        self.embedding_choices = self.define_embedding_choices()

    def load_environment_variables(self, env_path: str) -> None:
        """Load environment variables from a .env file."""
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
        """Retrieve the Pinecone API key."""
        return self.pinecone_api_key

    def get_openai_api_key(self) -> str:
        """Retrieve the OpenAI API key."""
        return self.openai_api_key

    def define_llm_choices(self) -> Dict[str, Any]:
        """Define available choices for language models (LLMs)."""
        return {
            "gpt": "gpt-4o",
            "ollama": [
                "llama3.1:8b-instruct-q5_K_M",
                "llama3.2:latest",
                "mistral-nemo:12b-instruct-2407-q5_K_M"
            ]
        }

    def define_embedding_choices(self) -> Dict[str, List[str]]:
        """Define available choices for embedding models."""
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
        """Validate a user selection against available choices."""
        if selection not in choices:
            raise ValueError(f"Invalid selection: {selection}. Available choices are: {choices}")

    def load_model(self, selected_llm_type: str, selected_llm: str, resource_manager) -> Any:
        """Load and initialize a language model."""
        try:
            if selected_llm_type == "gpt":
                return ChatOpenAI(openai_api_key=self.openai_api_key)
            else:
                return ChatOllama(model=selected_llm, **resource_manager)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def load_embeddings(self, selected_embedding_scheme: str, selected_embedding_model: str) -> Any:
        """Load and initialize embedding model."""
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
        """Determine embedding dimensions using sample text."""
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
        """Validate and load the selected language model and embedding model."""
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
