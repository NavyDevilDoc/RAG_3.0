# QuestionAnswerer.py
import time
import json
from typing import Dict, List, Optional
from ResponseSelector import ResponseSelector


class QuestionAnswerer:
    def __init__(self, chain, scoring_metric, embeddings, ground_truth_path: Optional[str] = None):
        """Initialize the QuestionAnswerer."""
        self.chain = chain
        self.scoring_metric = scoring_metric
        self.embeddings = embeddings
        self.ground_truth_path = ground_truth_path
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, str]:
        """Load ground truth data from JSON file."""
        if not self.ground_truth_path:
            return {}
            
        try:
            with open(self.ground_truth_path, "r") as file:
                data = json.load(file)
            print(f"Ground truth loaded from {self.ground_truth_path}")
            return data
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return {}


    def answer_questions(self, questions: List[str], datastore, use_ground_truth: bool = False, num_responses: int = 3):
        """Answer questions and return structured responses."""
        results = []
        start_time = time.time()
        selector = ResponseSelector()

        for question in questions:
            print(f"Processing Question: {question}\n")

            try: 
                print(f"Attempting to embed question: {question}")

                # Retrieve relevant documents
                try: 
                    retrieved_results = datastore.similarity_search_with_score(query=question, k=10)
                    print(f"Successfully retrieved {len(retrieved_results)} documents")
                except Exception as e:
                    print(f"Error in similarity search: {str(e)}")
                    continue
                scored_documents = self.scoring_metric.compute_relevance_score(question, [result[0].page_content for result in retrieved_results])
                top_documents = [doc[0] for doc in scored_documents]
                references = [doc[0].metadata.get('source', '') for doc in retrieved_results[:3]]

                # Generate responses
                responses = [self.chain.invoke({"question": question, "context": top_documents}) 
                            for _ in range(num_responses)]

                # Select best response and get confidence
                best_response = selector.select_best_response(question, responses)
                ranked_responses = selector.rank_responses(question, responses)
                confidence_score = ranked_responses[0][1]

                # Calculate quality scores using ground truth if available
                expected_answer = self.ground_truth.get(question) if use_ground_truth else None
                quality_scores = self.scoring_metric.compute_response_quality_score(
                    best_response, 
                    expected_answer
                )

                results.append({
                    'question': question,
                    'answer': best_response,
                    'confidence': confidence_score,
                    'references': references,
                    'quality_scores': quality_scores,
                    'processing_time': time.time() - start_time,
                    'ground_truth_used': bool(expected_answer)
                })

                print(f"Confidence Score: {confidence_score:.2f}")
                print(f"Quality Scores: {quality_scores}\n")

            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                continue

        return results