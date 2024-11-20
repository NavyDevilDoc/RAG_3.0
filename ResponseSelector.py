# ResponseSelector.py

# ResponseSelector ranks and selects the most relevant response from a list of candidates.
# It uses semantic similarity and confidence metrics to evaluate responses.

from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class ResponseSelector:
    REASONING_MARKERS = ['because', 'therefore', 'however']
    RELEVANCE_WEIGHT = 0.7
    CONFIDENCE_WEIGHT = 0.3

    # IMPORTANT: THE CONSTRUCTOR NEEDS TO BE UPDATED SO THAT THE model_name PARAMETER IS NOT HARD-CODED AND MATCHES THE USER'S SELECTION
    def __init__(self, model_name: str = "all-mpnet-base-v2", top_k: int = 5):
        """Initialize response selector with embedding model and top_k parameter.
        
        Args:
            model_name (str): Name of sentence transformer model for embeddings.
            top_k (int): Number of top responses to return when ranking. Must be > 0.
        
        Raises:
            ValueError: If the model fails to load or top_k is not a positive integer.
        """
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError(f"Error loading embedding model '{model_name}': {e}")
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        
        self.top_k = top_k

    def _get_relevance_score(self, question: str, response: str) -> float:
        """Calculate semantic similarity score between question and response.
        
        Args:
            question (str): Input question text.
            response (str): Candidate response text.
            
        Returns:
            float: Cosine similarity score between question and response embeddings (0 to 1).
        """
        if not question or not response:
            return 0.0
        
        q_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        r_embedding = self.embedding_model.encode(response, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(q_embedding, r_embedding))

    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response characteristics.
        
        Args:
            response (str): Response text to analyze.
            
        Returns:
            float: Confidence score based on length, specificity, and structure.
        
        Factors:
            - Length: Normalized by an expected length (100 words).
            - Specificity: Ratio of unique words to total words.
            - Structure: Presence of reasoning/transition markers (e.g., "because", "therefore").
        """
        if not response.strip():
            return 0.0

        factors = {
            'length': min(len(response.split()) / 100, 1.0),
            'specificity': len(set(response.split())) / len(response.split()),
            'structure': 1.0 if any(marker in response.lower() for marker in self.REASONING_MARKERS) else 0.7
        }
        return np.mean(list(factors.values()))

    def rank_responses(self, question: str, responses: List[str]) -> List[Tuple[str, float]]:
        """Rank responses by combining relevance and confidence scores.
        
        Args:
            question (str): Input question.
            responses (List[str]): List of candidate responses.
            
        Returns:
            List[Tuple[str, float]]: Ranked responses with scores, limited to top_k.
        
        Implementation:
            1. Calculate relevance and confidence for each response.
            2. Combine scores with weighted average (RELEVANCE_WEIGHT and CONFIDENCE_WEIGHT).
            3. Sort by final score and return the top_k responses.
        
        Raises:
            ValueError: If responses list is empty.
        """
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
        """Select single best response from candidates.
        
        Args:
            question (str): Input question.
            responses (List[str]): List of candidate responses.
            
        Returns:
            str: Best response based on ranking. Returns a default message if no responses.
        """
        ranked_responses = self.rank_responses(question, responses)
        return ranked_responses[0][0] if ranked_responses else "No suitable response found."
