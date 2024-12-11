# LLMQueryManager.py
import os
import json
import time
from tqdm import tqdm
from datetime import datetime
from typing import Any
from enum import Enum
from ModelManager import ModelManager
from langchain_core.output_parsers import StrOutputParser


class LLMType(Enum):
    """Available LLM types."""
    GPT = 'gpt'
    OLLAMA = 'ollama'


class LLMQueryManager:
    """Direct LLM query interface without RAG."""
    def __init__(
        self,
        env_path: str,
        json_path: str,
        llm_type: str,
        embedding_type: str,
        llm_model: str,
        embedding_model: str,
        debug_mode: bool = False,
        
    ):
        """Initialize LLM interface."""
        self.model_manager = ModelManager(env_path)
        self.json_path = json_path
        self.parser = StrOutputParser()
        self.llm_type = llm_type
        self.embedding_type = embedding_type
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.debug_mode = debug_mode
        self.model = self._initialize_model()
        self.conversation_history = []
        self.history_file = os.path.join(os.path.dirname(json_path), "conversation_history.json")
        self.conversation_history = self.load_history()
        self.max_history_length = 50  
        self.max_tokens_per_message = 1000 
        self.max_context_tokens = 8000  
        
        if self.llm_type.lower() == 'ollama':
            self._initialize_ollama_model_with_progress()
        else:
            self._initialize_model()

    def _initialize_ollama_model_with_progress(self):
        """Initialize Ollama model with a progress bar."""
        with tqdm(total=100, desc="Loading Ollama Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for i in range(10):
                time.sleep(0.5)  # Simulate loading time
                pbar.update(10)
        
        self._initialize_model()


    def _initialize_model(self) -> Any:
        """Initialize the language model."""
        config = {
            "selected_llm_type": self.llm_type,
            "selected_embedding_scheme": self.embedding_type  
        }
        
        model, _, _, self.selected_llm, self.selected_embedding_model = self.model_manager.validate_and_load_models(
            config=config,
            select_llm=self.llm_model,
            select_embed=self.embedding_model,
            resource_manager={}
        )
        return model


    def load_history(self) -> list:
        """Load conversation history from JSON file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
        return []


    def save_history(self):
        """Save conversation history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")


    def add_to_history(self, role: str, content: str):
        """Add message with history management."""
        # Trim history if exceeding limits
        while (len(self.conversation_history) >= self.max_history_length or
               self._estimate_total_tokens() >= self.max_context_tokens):
            self.conversation_history.pop(0)  # Remove oldest message
            
        # Add new message
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save_history()
        
        
    def _estimate_total_tokens(self) -> int:
        """Rough token count estimation."""
        return sum(len(msg["content"].split()) * 1.3 
                  for msg in self.conversation_history)
        

    def get_conversation_context(self) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])


    def ask(self, question: str, use_history: bool = True) -> str:
        try:
            self.add_to_history("user", question)
            
            # Include history in prompt if enabled
            prompt = (
                self.get_conversation_context() if use_history and self.conversation_history 
                else question
            )
            
            chain = self.model | self.parser
            response = chain.invoke(prompt)
            
            self.add_to_history("assistant", response)
            
            if self.debug_mode:
                print(f"\nModel: {self.selected_llm}")
                print(f"History length: {len(self.conversation_history)}")
                
            return response
            
        except Exception as e:
            print(f"Error in LLM query: {e}")
            return f"Error processing question: {e}"