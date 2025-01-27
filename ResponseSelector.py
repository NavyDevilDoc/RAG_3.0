from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
import numpy as np
from ScoringMetric import ScoringMetric

class ResponseSelector:
    REASONING_MARKERS = ['because', 'therefore', 'however']
    RELEVANCE_WEIGHT = 0.7
    CONFIDENCE_WEIGHT = 0.3

    def __init__(self, model_name: str = "all-mpnet-base-v2", top_k: int = 5):
        """Initialize response selector with embedding model and top_k parameter."""
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError(f"Error loading embedding model '{model_name}': {e}")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        self.top_k = top_k
        self.scoring_metric = ScoringMetric(model_name, 'sentence_transformer')


    def _get_relevance_score(self, question: str, response: str) -> float:
        """Calculate semantic similarity score between question and response."""
        if not question or not response:
            return 0.0
        try:
            q_embedding = self.embedding_model.encode(question)
            r_embedding = self.embedding_model.encode(response)
            return float(util.pytorch_cos_sim(q_embedding, r_embedding))
        except Exception as e:
            print(f"Error calculating relevance score: {e}")
            return 0.0


    def rank_responses(self, question: str, responses: List[str]) -> List[Tuple[str, float]]:
        """Rank responses by combining relevance and confidence scores."""
        if not responses:
            raise ValueError("The 'responses' list cannot be empty.")
        scored_responses = []
        for response in responses:
            try:
                relevance = self._get_relevance_score(question, response)
                confidence = self.scoring_metric.calculate_confidence(response, question)
                final_score = self.RELEVANCE_WEIGHT * relevance + self.CONFIDENCE_WEIGHT * confidence
                scored_responses.append((response, final_score))
            except Exception as e:
                print(f"Error ranking response: {e}")
        # Sort and return only the top_k responses
        try:
            return sorted(scored_responses, key=lambda x: x[1], reverse=True)[:self.top_k]
        except Exception as e:
            print(f"Error sorting responses: {e}")
            return []


    def select_best_response(self, question: str, responses: List[str]) -> str:
        """Select single best response from candidates."""
        try:
            ranked_responses = self.rank_responses(question, responses)
            return ranked_responses[0][0] if ranked_responses else "No suitable response found."
        except Exception as e:
            print(f"Error selecting best response: {e}")
            return "No suitable response found."