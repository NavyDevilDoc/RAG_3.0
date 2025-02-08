# QuestionAnswerer.py
import time
import json
from typing import Dict, List, Optional
from ResponseSelector import ResponseSelector
from ScoringMetric import ScoringMetric
import numpy as np


class QuestionAnswerer:
    def __init__(self, 
                 chain, 
                 embedding_model, 
                 embedding_type, 
                 ground_truth_path: Optional[str] = None, 
                 use_reranking: bool = True, 
                 save_outputs: bool = False, 
                 output_file_path: str = "re-ranking_test_outputs.txt",
                 num_responses: int = 5
                 ):
        """Initialize the QuestionAnswerer."""
        self.chain = chain
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.ground_truth_path = ground_truth_path
        self.use_reranking = use_reranking
        self.save_outputs = save_outputs
        self.output_file_path = output_file_path
        self.num_responses = num_responses
        self.ground_truth = self._load_ground_truth()


    def _load_ground_truth(self) -> Dict[str, str]:
        """Load ground truth data from JSON file."""
        if not self.ground_truth_path:
            return {}
        try:
            with open(self.ground_truth_path, "r") as file:
                data = json.load(file)
            #print(f"Ground truth loaded from {self.ground_truth_path}")
            return data
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return {}


    def answer_questions(self, questions: List[str], datastore, use_ground_truth: bool = False):
        """Answer questions and return structured responses."""
        """ Experimenting with k, top percentage, and re-ranked documents going to the response selector """
        results = []
        start_time = time.time()
        selector = ResponseSelector(use_reranking=self.use_reranking, save_outputs=self.save_outputs, output_file_path=self.output_file_path)
        scoring_metric = ScoringMetric(self.embedding_model, self.embedding_type)

        for question in questions:
            try:
                # Use Pinecone's similarity search to retrieve the top k documents
                retrieved_results = datastore.similarity_search_with_score(query=question, k=20)
                print(f"\n-----Retrieved Results-----")
                for result in retrieved_results:
                    page_number = result[0].metadata.get('page', 'Unknown')
                    print(f"Document #{page_number}: {result[0].page_content}... Score: {result[1]}")


                # Select a percentage scored documents for re-ranking
                top_percentage = 0.4
                top_percentage_index = int(np.ceil(len(retrieved_results) * top_percentage))
                documents_for_reranking = [result[0].page_content for result in retrieved_results[:top_percentage_index]]

                # Re-rank documents
                reranked_documents = selector.rerank_documents(question, documents_for_reranking)
                print(f"\n-----Re-ranked Documents-----")
                for doc, score in reranked_documents:
                    print(f"Document: {doc[:100]}... Score: {score}")

                # Use top n re-ranked documents to generate responses
                top_documents = [doc for doc, _ in reranked_documents[:3]]
                print(f"\n-----Top Documents for Response Generation-----")
                for doc in top_documents:
                    print(f"Document: {doc[:100]}...")

                references = [result[0].metadata.get('source', '') for result in retrieved_results[:3]]

                # Generate LLM responses
                responses = [self.chain.invoke({"question": question, "context": top_documents}) for _ in range(self.num_responses)]

                # Select best response and get confidence
                ranked_responses = selector.rank_responses(question, responses)
                best_response = selector.select_best_response(ranked_responses) 
                confidence_score = ranked_responses[0][1]

                # Calculate quality scores using ground truth if available
                expected_answer = self.ground_truth.get(question) if use_ground_truth else None
                quality_scores = scoring_metric.compute_response_quality_score(best_response, expected_answer)

                results.append({
                    'question': question,
                    'answer': best_response,
                    'confidence': confidence_score,
                    'references': references,
                    'quality_scores': quality_scores,
                    'processing_time': time.time() - start_time,
                    'ground_truth_used': bool(expected_answer)
                })

                print(f"\nQuality Scores: {quality_scores}\n")

                # Save results to DataFrame
                if self.save_outputs:
                    original_results = [(result[0].page_content, result[1]) for result in retrieved_results]
                    re_ranked_results = [(doc, score) for doc, score in reranked_documents]
                    response_results = [(response, score) for response, score in ranked_responses]
                    df = selector.save_results_to_dataframe(original_results, re_ranked_results, response_results)
                    filename = "dataframe_output.csv"
                    df.to_csv(filename, index=False)

            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                continue

        return results