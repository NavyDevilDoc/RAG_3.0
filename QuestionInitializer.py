# QuestionInitializer.py
from typing import List, Dict, Any
import time
import os
from ChainManager import ChainManager
from QuestionAnswerer import QuestionAnswerer
from TemplateManager import TemplateManager

class QuestionInitializer:
    """Manages question answering workflow including templates, ground truth, and pipeline execution."""
    
    def __init__(self,
                 datastore: Any,
                 model: Any,
                 embedding_model: Any,
                 embedding_type: Any,
                 template_path: str = "templates.json",
                 ground_truth_path: str = "ground_truth.json",
                 use_reranking: bool = True,
                 save_outputs: bool = True,
                 output_file_path: str = "re-ranking_test_outputs.txt",
                 num_responses: int = 1
                 ):
        """Initialize question processor with models and paths."""
        self.datastore = datastore
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.template_path = template_path
        self.ground_truth_path = ground_truth_path
        self.use_reranking = use_reranking
        self.save_outputs = save_outputs
        self.output_file_path = output_file_path
        self.num_responses = num_responses

        # Initialize components
        self.template_manager = TemplateManager(template_path)
        self.chain_manager = None
        self.question_answerer = None


    def _initialize_pipeline(self, template: str):
        """Initialize chain manager and question answerer components."""
        self.chain_manager = ChainManager(self.datastore, self.model, template)
        chain = self.chain_manager.setup_chain()
        self.question_answerer = QuestionAnswerer(chain, 
                                                self.embedding_model,
                                                self.embedding_type,
                                                self.ground_truth_path,
                                                use_reranking=self.use_reranking,
                                                save_outputs=self.save_outputs,
                                                output_file_path=self.output_file_path,
                                                num_responses=self.num_responses)

    def _load_template(self, template_name: str = "default") -> str:
        """Load specific template."""
        template = self.template_manager.get_template(template_name)
        #print(f"Loaded template: {template}")
        return template

    def process_questions(self,
                        questions: List[str],
                        use_ground_truth: bool = False,
                        template_name: str = "default") -> List[Dict]:
        """Process questions and return execution time."""
        try:
            # Load template
            template = self._load_template(template_name)
            
            # Initialize pipeline componentsS
            self._initialize_pipeline(template)
            
            # Process questions and measure time
            start_time = time.time()
            results = self.question_answerer.answer_questions(
                questions,
                self.datastore,
                use_ground_truth
            )
            processing_time = time.time() - start_time
            
            return results, processing_time
            
        except Exception as e:
            raise RuntimeError(f"Failed to process questions: {e}")