# QuestionAnswerer.py
import time
import json
from typing import Dict, List, Optional
from ResponseSelector import ResponseSelector
from ScoringMetric import ScoringMetric
import os


class QuestionAnswerer:
    def __init__(self, chain, embedding_model, embedding_type, ground_truth_path: Optional[str] = None, use_reranking: bool = True, save_outputs: bool = False, output_file_path: str = "re-ranking_test_outputs.txt"):
        """Initialize the QuestionAnswerer."""
        self.chain = chain
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.ground_truth_path = ground_truth_path
        self.use_reranking = use_reranking
        self.save_outputs = save_outputs
        self.output_file_path = output_file_path
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


    def answer_questions(self, questions: List[str], datastore, use_ground_truth: bool = False, num_responses: int = 1):
        """Answer questions and return structured responses."""
        results = []
        start_time = time.time()
        selector = ResponseSelector(use_reranking=self.use_reranking, save_outputs=self.save_outputs, output_file_path=self.output_file_path)
        scoring_metric = ScoringMetric(self.embedding_model, self.embedding_type)

        for question in questions:
            try:
                # Retrieve relevant documents
                retrieved_results = datastore.similarity_search_with_score(query=question, k=20)
                print(f"-----Retrieved Results-----")
                for result in retrieved_results:
                    print(f"Document: {result[0].page_content[:100]}... Score: {result[1]}")

                # Score documents
                scored_documents = scoring_metric.compute_relevance_score(question, [result[0].page_content for result in retrieved_results])
                scored_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
                print(f"-----Scored Documents-----")
                for doc, score in scored_documents:
                    print(f"Document: {doc[:100]}... Score: {score}")

                # Select top 50% of scored documents for re-ranking
                top_50_percent_index = len(scored_documents) // 2
                documents_for_reranking = [doc for doc, _ in scored_documents[:top_50_percent_index]]
                print(f"-----Documents for Re-ranking-----")
                for doc in documents_for_reranking:
                    print(f"Document: {doc[:100]}...")

                # Re-rank documents
                reranked_documents = selector.rerank_documents(question, documents_for_reranking)
                print(f"-----Re-ranked Documents-----")
                for doc, score in reranked_documents:
                    print(f"Document: {doc[:100]}... Score: {score}")

                # Use top 5 re-ranked documents to generate responses
                top_documents = [doc for doc, _ in reranked_documents[:5]]
                print(f"-----Top Documents for Response Generation-----")
                for doc in top_documents:
                    print(f"Document: {doc[:100]}...")

                references = [result[0].metadata.get('source', '') for result in retrieved_results[:3]]

                # Generate responses
                responses = [self.chain.invoke({"question": question, "context": top_documents}) for _ in range(num_responses)]

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

                print(f"\nConfidence Score: {confidence_score:.2f}")
                print(f"Quality Scores: {quality_scores}\n")

                # Save results to DataFrame
                if self.save_outputs:
                    original_results = [(result[0].page_content, result[1]) for result in retrieved_results]
                    re_ranked_results = [(response, score) for response, score in ranked_responses]
                    df = selector.save_results_to_dataframe(original_results, re_ranked_results)
                    filename = "dataframe_output.csv"
                    df.to_csv(filename, index=False)
                    #print(df)

            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                continue

        return results