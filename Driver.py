# Driver.py - Driver class for RAG model execution

import os
from RAGInitializer import EmbeddingType, RAGConfig, initialize_rag_components
from CombinedProcessor import CombinedProcessor, ChunkingMethod
from DatastoreInitializer import StorageType
from QuestionInitializer import QuestionInitializer
from TextPreprocessor import TextPreprocessor
from ResponseFormatter import QAResponse
from LLMQueryManager import LLMQueryManager, LLMType
from typing import List
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Driver:
    def __init__(
        self,
        env_path: str,
        json_path: str,
        doc_input: List[str] = None,
        llm_type: str = 'OLLAMA',
        embedding_type: str = 'SENTENCE_TRANSFORMER',
        llm_model: str = 'llama3.2:latest', 
        embedding_model: str = 'all-mpnet-base-v2', 
        doc_name: str = 'test-index',
        chunking_method: str = 'HIERARCHICAL',
        storage_type: str = 'PINECONE_EXISTING',
        template_name: str = 'default',
        use_ground_truth: bool = False,
        debug_mode: bool = False,
        mode: str = 'rag',
        process_questions: bool = True
    ):
        """Initialize RAG driver based on mode."""
        self.mode = mode.lower()
        self.should_process_questions = process_questions
        
        # For all modes
        self.env_path = env_path
        self.json_path = json_path
        self.llm_type = llm_type
        self.embedding_type = embedding_type
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.debug_mode = debug_mode
        
        # Initialize LLM query manager for direct queries
        if self.mode == 'llm':
            if self.llm_type.lower() == 'ollama':
                self._initialize_ollama_model_with_progress()
            else:
                self.llm_query = LLMQueryManager(
                    env_path=env_path,
                    json_path=json_path,
                    llm_type=llm_type,
                    llm_model=llm_model,
                    embedding_type=embedding_type,
                    embedding_model=embedding_model,
                    debug_mode=debug_mode
                )
            return  # Early exit - don't initialize RAG components
            
        # Only initialize RAG components if in RAG mode
        self.doc_input = doc_input
        self.embedding_type = embedding_type
        self.embedding_index = embedding_model
        self.doc_name = doc_name
        self.chunking_method = chunking_method
        self.storage_type = storage_type
        self.template_name = template_name
        self.use_ground_truth = use_ground_truth
        self.mode = mode
        
        # Initialize instance variables
        self.model = None
        self.embeddings = None
        self.dimensions = None
        self.selected_llm = None
        self.selected_embedding_model = None
        self.model_manager = None
        self.datastore = None
        
        
        # Initial setup for RAG mode
        self.setup()


    def _initialize_ollama_model_with_progress(self):
        """Initialize Ollama model with a progress bar."""
        with tqdm(total=100, desc="Loading Ollama Model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for i in range(10):
                time.sleep(0.5)  # Simulate loading time
                pbar.update(10)
        
        self.llm_query = LLMQueryManager(
            env_path=self.env_path,
            json_path=self.json_path,
            llm_type=self.llm_type,
            llm_model=self.llm_model,
            embedding_type=self.embedding_type,
            embedding_model=self.embedding_model,
            debug_mode=self.debug_mode
        )


    @staticmethod
    def _convert_llm_type(llm_type_str: str) -> LLMType:
        """Convert string to LLMType enum case-insensitively"""
        try:
            return getattr(LLMType, llm_type_str.upper())
        except AttributeError:
            raise ValueError(f"Invalid LLM type: {llm_type_str}. Must be one of: {[t.name for t in LLMType]}")

    @staticmethod
    def _convert_embedding_type(embedding_type_str: str) -> EmbeddingType:
        """Convert string to EmbeddingType enum case-insensitively"""
        try:
            return getattr(EmbeddingType, embedding_type_str.upper())
        except AttributeError:
            raise ValueError(f"Invalid embedding type: {embedding_type_str}. Must be one of: {[t.name for t in EmbeddingType]}")
    
    @staticmethod
    def _convert_chunking_method(chunking_method_str: str) -> ChunkingMethod:
        return getattr(ChunkingMethod, chunking_method_str)
    
    @staticmethod
    def _convert_storage_type(storage_type_str: str) -> StorageType:
        return getattr(StorageType, storage_type_str)
    
    @staticmethod
    def find_pdfs_in_folder(folder_path: str) -> list:
        """Find all PDF files in the specified folder and return their file paths."""
        pdf_files = []
        for root, dirs, files in os.walk(folder_path):
            print(f"Found {len(files)} files in {root}")
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files
    

    def setup(self):
        """Initialize RAG components and process documents."""
        # Initialize RAG components
        config = RAGConfig(
            env_path=self.env_path,
            llm_type=self._convert_llm_type(self.llm_type),
            embedding_type=self._convert_embedding_type(self.embedding_type),
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            mode=self.mode
        )
        
        components = initialize_rag_components(config)
        if len(components) == 6:  # Full RAG mode
            (self.model, self.embeddings, self.dimensions, 
            self.selected_llm, self.selected_embedding_model, 
            self.model_manager) = components
        elif len(components) == 4:  # LLM mode
            (self.model, self.embeddings, self.selected_llm, 
            self.selected_embedding_model) = components
            self.dimensions = None  # Set default
            self.model_manager = None  # Set default
        else:
            raise ValueError(f"Unexpected number of components: {len(components)}")
        
        if self.model and self.embeddings and self.dimensions:
            print("\nRAG components successfully initialized.")
            print(f"Model: {self.selected_llm}")
            print(f"Embeddings: {self.selected_embedding_model}")
            
        # Process documents
        processor = CombinedProcessor(
            doc_name=self.doc_name,
            model_manager=self.model_manager,
            embedding_model=self.selected_embedding_model,
            embeddings=self.embeddings,
            dimensions=self.dimensions,
            chunking_method=self._convert_chunking_method(self.chunking_method),
            storage_type=self._convert_storage_type(self.storage_type),
            model_name=self.selected_llm
        )
        
        self.datastore = processor.process_and_store(self.doc_input)
        
    def process_questions(self, questions: List[str]) -> List[QAResponse]:
        """Process questions and return QAResponse objects."""
        if not self.datastore:
            raise RuntimeError("Setup must be called before processing questions")
        
        if not questions:
            raise ValueError("Questions list cannot be empty")
                
        processor = QuestionInitializer(
            datastore=self.datastore,
            model=self.model,
            embedding_model=self.selected_embedding_model
        )
        
        try:
            # Get responses and handle tuple return
            result = processor.process_questions(
                questions=questions,
                template_name=self.template_name,     
                use_ground_truth=self.use_ground_truth[0] if isinstance(self.use_ground_truth, tuple) else self.use_ground_truth
            )
            
            # Unpack tuple if needed
            if isinstance(result, tuple):
                results = result[0]  # Get just the results, ignore processing time
            else:
                results = result
                
            # Ensure results is a list
            if not isinstance(results, list):
                print(f"Warning: Invalid results format after unpacking: {type(results)}")
                return []
                
            responses = []
            text_processor = TextPreprocessor()
            # Map each result to its corresponding question
            for question, result in zip(questions, results):
                if not isinstance(result, dict):
                    print(f"Warning: Invalid result format: {type(result)}")
                    continue
                    
                try:
                    response = QAResponse(
                        question=question,
                        answer=result['answer'],
                        confidence=result.get('confidence', 0.0),
                        references=result.get('references', []),
                        metadata={
                            'model': self.selected_llm,
                            'embeddings': self.selected_embedding_model,
                            'template': self.template_name,
                            'chunking_method': self.chunking_method,
                            'quality_scores': result.get('quality_scores', {})
                        }
                    )
                    # Add token count to the answer
                    response.answer += f"\n\nToken Count: {text_processor.count_tokens(response.answer)}"
                    responses.append(response)
                except KeyError as e:
                    print(f"Warning: Missing required field in result: {e}")
                    continue
                    
            return responses
            
        except Exception as e:
            raise RuntimeError(f"Error processing questions: {e}")


    def process_query(self, query: str, use_history: bool = True) -> str:
        """Process query with conversation history support."""
        text_processor = TextPreprocessor()
        
        if self.mode == 'llm':
            if not self.llm_query:
                raise RuntimeError("LLM mode not initialized")
            
            response = self.llm_query.ask(query, use_history=use_history)
            qa_response = QAResponse(question=query, answer=response, confidence=0.0)
            formatted_response = text_processor.format_text(response)
            token_count = text_processor.count_tokens(formatted_response)
            return f"{formatted_response}\n\nToken Count: {token_count}"
        else:
            # RAG mode remains unchanged
            qa_response = self.process_questions([query])[0]
            formatted_response = text_processor.format_text(qa_response.answer)
            token_count = text_processor.count_tokens(formatted_response)
            return f"{formatted_response}\n\nToken Count: {token_count}"


    def clear_conversation_history(self):
        """Clear the conversation history in LLM mode."""
        if self.mode == 'llm' and self.llm_query:
            self.llm_query.conversation_history = []


    def get_conversation_history(self):
        """Get the current conversation history in LLM mode."""
        if self.mode == 'llm' and self.llm_query:
            return self.llm_query.conversation_history
        return []


    def run(self, questions: List[str]) -> List[str]:
        """Run complete RAG pipeline."""
        if not self.should_process_questions:
            print("Question processing disabled - datastore operations completed")
            return []
        return self.process_questions(questions)