# QuestionInitializer.py
from typing import List, Dict, Any
import time
from ScoringMetric import ScoringMetric
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
                 ground_truth_path: str = "ground_truth.json"
                 ):
        """Initialize question processor with models and paths."""
        self.datastore = datastore
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.template_path = template_path
        self.ground_truth_path = ground_truth_path
        
        # Initialize components
        self.template_manager = TemplateManager(template_path)
        self.scoring_metric = ScoringMetric(embedding_model, embedding_type)
        self.chain_manager = None
        self.question_answerer = None

    def _initialize_pipeline(self, template: str):
        """Initialize chain manager and question answerer components."""
        self.chain_manager = ChainManager(self.datastore, self.model, template)
        chain = self.chain_manager.setup_chain()
        self.question_answerer = QuestionAnswerer(chain, 
                                                self.scoring_metric,
                                                self.embedding_model,
                                                self.ground_truth_path)

    def _load_template(self, template_name: str = "default") -> str:
        """Load specific template."""
        template = self.template_manager.get_template(template_name)
        print(f"Loaded template: {template}")
        return template

    def process_questions(self,
                        questions: List[str],
                        use_ground_truth: bool = False,
                        template_name: str = "default",
                        num_responses: int = 3) -> List[Dict]:
        """Process questions and return execution time."""
        try:
            # Load template
            template = self._load_template(template_name)
            
            # Initialize pipeline components
            self._initialize_pipeline(template)
            print(f"Processing questions: {questions}")  # Debugging statement
            
            # Process questions and measure time
            start_time = time.time()
            results = self.question_answerer.answer_questions(
                questions,
                self.datastore,
                use_ground_truth,
                num_responses
            )
            processing_time = time.time() - start_time
            
            return results, processing_time
            
        except Exception as e:
            raise RuntimeError(f"Failed to process questions: {e}")