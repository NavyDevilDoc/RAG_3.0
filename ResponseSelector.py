# ResponseSelector.py

# ResponseSelector ranks and selects the most relevant response from a list of candidates.
# It uses semantic similarity and confidence metrics to evaluate responses.

from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple
import numpy as np

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


    def _get_relevance_score(self, question: str, response: str) -> float:
        """Calculate semantic similarity score between question and response."""
        
        if not question or not response:
            return 0.0
        q_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        r_embedding = self.embedding_model.encode(response, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(q_embedding, r_embedding))


    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response characteristics."""
        if not response.strip():
            return 0.0
        factors = {
            'length': min(len(response.split()) / 100, 1.0),
            'specificity': len(set(response.split())) / len(response.split()),
            'structure': 1.0 if any(marker in response.lower() for marker in self.REASONING_MARKERS) else 0.7
        }
        return np.mean(list(factors.values()))


    def rank_responses(self, question: str, responses: List[str]) -> List[Tuple[str, float]]:
        """Rank responses by combining relevance and confidence scores."""
        
        if not responses:
            raise ValueError("The 'responses' list cannot be empty.")

        scored_responses = []
        for response in responses:
            relevance = self._get_relevance_score(question, response)
            confidence = self._calculate_confidence(response)
            final_score = self.RELEVANCE_WEIGHT * relevance + self.CONFIDENCE_WEIGHT * confidence
            scored_responses.append((response, final_score))
        # Sort and return only the top_k responses
        return sorted(scored_responses, key=lambda x: x[1], reverse=True)[:self.top_k]


    def select_best_response(self, question: str, responses: List[str]) -> str:
        """Select single best response from candidates."""
        ranked_responses = self.rank_responses(question, responses)
        return ranked_responses[0][0] if ranked_responses else "No suitable response found."
