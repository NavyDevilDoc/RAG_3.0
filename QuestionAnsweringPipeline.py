from ChainManager import ChainManager
from QuestionAnswerer import QuestionAnswerer

class QuestionAnsweringPipeline:
    def __init__(self, datastore, model, template, scoring_metric, embeddings):
        """Initialize the pipeline with core components."""
        self.chain_manager = ChainManager(datastore, model, template)
        self.chain = self.chain_manager.setup_chain()
        self.question_answerer = QuestionAnswerer(self.chain, 
                                                  scoring_metric, 
                                                  embeddings)

    def answer_questions(self, questions, ground_truth=None, num_responses=3):
        """Answer questions using the pipeline."""
        return self.question_answerer.answer_questions(questions, 
                                                       self.chain_manager.datastore, 
                                                       ground_truth, 
                                                       num_responses)
