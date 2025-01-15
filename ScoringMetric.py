from sentence_transformers import util, SentenceTransformer
from langchain_openai.embeddings import OpenAIEmbeddings
import evaluate
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class ScoringMetric:
    def __init__(self, embedding_model, embedding_type):
        """Initialize the ScoringMetric class with an embedding model for relevance scoring."""
        self.embedding_model = embedding_model
        self.embedding_type = embedding_type
        self.rouge = evaluate.load('rouge')  # Load ROUGE from Hugging Face evaluate library
        self.bleu = evaluate.load('bleu')    # Load BLEU from Hugging Face evaluate library

    def compute_relevance_score(self, query: str, retrieved_documents: List[str]) -> List[Tuple[str, float]]:
        """Compute relevance scores for retrieved documents."""

        try:
            print(f"Computing relevance for query: {query}")
            if self.embedding_type == 'sentence_transformer':
                self.embedding_model = SentenceTransformer(self.embedding_model)
                query_embedding = self.embedding_model.encode(query)
                scored_documents = [
                    (doc, float(util.pytorch_cos_sim(query_embedding, self.embedding_model.encode(doc))))
                    for doc in retrieved_documents
                ]
            elif self.embedding_type == 'gpt':
                self.embedding_model = OpenAIEmbeddings(model=self.embedding_model)
                query_embedding = self.embedding_model.embed_query(query)
                scored_documents = [
                    (doc, np.dot(query_embedding, self.embedding_model.embed_documents([doc])[0]))
                    for doc in retrieved_documents
                ]
            return sorted(scored_documents, key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"Error in relevance scoring: {str(e)}")
            return []

    def compute_response_quality_score(self, response: str, ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Compute quality score for the generated response using ROUGE, BLEU, and confidence."""
        scores = {}
        # Calculate ROUGE score if ground truth is provided
        if ground_truth:
            rouge_scores = self.rouge.compute(predictions=[response], references=[ground_truth])
            scores.update(rouge_scores)
        # Calculate BLEU score if ground truth is provided
        if ground_truth:
            bleu_scores = self.bleu.compute(predictions=[response.split()], references=[[ground_truth.split()]])
            scores.update(bleu_scores)
        # Calculate confidence score
        confidence_score = self.calculate_confidence(response)
        scores['confidence'] = confidence_score
        return scores

    def calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response characteristics."""
        if not response.strip():
            return 0.0
        factors = {
            'length': min(len(response.split()) / 100, 1.0),
            'specificity': len(set(response.split())) / len(response.split()),
            'structure': 1.0 if any(marker in response.lower() for marker in ['because', 'therefore', 'however']) else 0.7
        }
        return np.mean(list(factors.values()))