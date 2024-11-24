# QuestionInitializer.py
from typing import List, Dict, Optional, Any
import time
from ScoringMetric import ScoringMetric
from QuestionAnsweringPipeline import QuestionAnsweringPipeline
from TemplateManager import TemplateManager
from GroundTruthManager import GroundTruthManager

class QuestionInitializer:
    """Manages question answering workflow including templates, ground truth, and pipeline execution."""
    
    def __init__(self,
                 datastore: Any,
                 model: Any,
                 embedding_model: Any,
                 template_path: str = "templates.json",
                 ground_truth_path: str = "ground_truth.json"):
        """Initialize question processor with models and paths."""
        self.datastore = datastore
        self.model = model
        self.embedding_model = embedding_model
        self.template_path = template_path
        self.ground_truth_path = ground_truth_path
        
        # Initialize components
        self.template_manager = TemplateManager(template_path)
        self.scoring_metric = ScoringMetric(embedding_model)
        
    def _load_template(self, template_name: str = "default") -> str:
        """Load specific template."""
        template = self.template_manager.get_template(template_name)
        print(f"Loaded template: {template}")
        return template
        
    def _get_ground_truth(self, 
                         questions: List[str], 
                         use_ground_truth: bool = False) -> Optional[Dict[str, str]]:
        """Retrieve ground truth answers if available."""
        if not use_ground_truth:
            return None
            
        try:
            ground_truth_manager = GroundTruthManager(self.ground_truth_path)
            return {q: ground_truth_manager.get_answer(q) for q in questions}
        except Exception as e:
            print(f"Warning: Failed to load ground truth: {e}")
            return None
            
    def process_questions(self,
                         questions: List[str],
                         use_ground_truth: bool = False,
                         template_name: str = "default") -> List[Dict]:
        """Process questions through pipeline and return execution time."""
        try:
            # Load template and ground truth
            template = self._load_template(template_name)
            ground_truth = self._get_ground_truth(questions, use_ground_truth)
            
            # Initialize and run pipeline
            pipeline = QuestionAnsweringPipeline(
                self.datastore,
                self.model,
                template,
                self.scoring_metric,
                self.embedding_model
            )
            
            # Process questions and measure time
            start_time = time.time()
            results = pipeline.answer_questions(questions, ground_truth)
            processing_time = time.time() - start_time
            
            return results, processing_time
            
        except Exception as e:
            raise RuntimeError(f"Failed to process questions: {e}")