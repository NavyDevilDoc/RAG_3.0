from sentence_transformers import util
import evaluate
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class ScoringMetric:
    def __init__(self, embedding_model):
        """Initialize the ScoringMetric class with an embedding model for relevance scoring."""
        self.embedding_model = embedding_model
        self.rouge = evaluate.load('rouge')  # Load ROUGE from Hugging Face evaluate library
        self.bleu = evaluate.load('bleu')    # Load BLEU from Hugging Face evaluate library

    def compute_relevance_score(self, query: str, retrieved_documents: List[str]) -> List[Tuple[str, float]]:
        """Compute relevance scores for retrieved documents."""
        try:
            if hasattr(self.embedding_model, 'encode'):  # For models like SentenceTransformer
                print(f"Computing relevance for query: {query}")
                if callable(getattr(self.embedding_model, 'encode', None)):
                    query_embedding = self.embedding_model.encode(query)
                else:
                    raise AttributeError("The embedding model does not have an 'encode' method.")
                print(f"Successfully embedded query")
                scored_documents = [
                    (doc, float(util.pytorch_cos_sim(query_embedding, self.embedding_model.encode(doc))))
                    for doc in retrieved_documents
                ]
            else:  # Assume OpenAI Embeddings API
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
            
            # Calculate BLEU score
            bleu_score = self.bleu.compute(
                predictions=[response],  # Single string as a list
                references=[[ground_truth]]  # Single string wrapped in a list of lists
            )
            scores['bleu'] = bleu_score['bleu']

        # Confidence metric, based on response length and specificity
        scores['confidence'] = self.calculate_confidence(response)
        
        return scores


    def calculate_confidence(self, response: str) -> float:
        """A custom confidence score based on response characteristics."""
        
        length_factor = min(len(response.split()) / 100, 1.0)
        specificity = len(set(response.split())) / len(response.split()) if response else 0
        return (length_factor + specificity) / 2