import time
from ResponseSelector import ResponseSelector
from TextPreprocessor import TextPreprocessor
from ScoringMetric import ScoringMetric

class QuestionAnswerer:
    def __init__(self, chain, scoring_metric, embeddings):
        """
        Initialize the QuestionAnswerer.
        
        Args:
            chain: The chain to use for question answering
            scoring_metric: Metric to evaluate response quality
            embeddings: Embedding model for text processing
        """
        self.chain = chain
        self.scoring_metric = scoring_metric
        self.embeddings = embeddings

    def answer_questions(self, questions, datastore, ground_truth=None, num_responses=3):
        """Answer a list of questions using the provided chain."""
        start_time = time.time()
        selector = ResponseSelector()
        all_quality_scores = []

        for question in questions:
            print(f"Processing Question: {question}\n")

            # Retrieve relevant documents
            retrieved_results = datastore.similarity_search_with_score(query=question, k=10)
            scored_documents = [(result[0].page_content, result[1]) for result in retrieved_results]
            top_documents = [doc[0] for doc in scored_documents]

            # Generate responses
            responses = [self.chain.invoke({"question": question, "context": top_documents}) for _ in range(num_responses)]

            # Select best response and rank responses
            best_response = selector.select_best_response(question, responses)
            ranked_responses = selector.rank_responses(question, responses)

            # Calculate quality scores
            expected_answer = ground_truth.get(question) if ground_truth else None
            quality_scores = self.scoring_metric.compute_response_quality_score(best_response, expected_answer)
            all_quality_scores.append(quality_scores)

            processor = TextPreprocessor()
            print(f"Best Response: {processor.format_text(best_response,100)}")
            print(f"Confidence Score: {ranked_responses[0][1]:.2f}")
            print(f"Quality Scores: {quality_scores}\n")

        # Calculate runtime
        total_time = time.time() - start_time
        return total_time
