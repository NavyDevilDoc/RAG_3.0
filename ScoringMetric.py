import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import util
import evaluate  # Hugging Face evaluate library for ROUGE and BLEU
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class ScoringMetric:
    def __init__(self, embedding_model):
        """Initialize the ScoringMetric class with an embedding model for relevance scoring.
        
        Args:
            embedding_model: Embedding model instance, e.g., OpenAIEmbeddings or SentenceTransformer.
        """
        self.embedding_model = embedding_model
        self.rouge = evaluate.load('rouge')  # Load ROUGE from Hugging Face evaluate library
        self.bleu = evaluate.load('bleu')    # Load BLEU from Hugging Face evaluate library

    def compute_relevance_score(self, query: str, retrieved_documents: List[str]) -> List[Tuple[str, float]]:
        """Compute relevance scores for retrieved documents.
        
        Args:
            query (str): Input query text.
            retrieved_documents (List[str]): List of retrieved document texts.
        
        Returns:
            List[Tuple[str, float]]: Documents with their relevance scores.
        """
        if hasattr(self.embedding_model, 'encode'):  # For models like SentenceTransformer
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            scored_documents = [
                (doc, float(util.pytorch_cos_sim(query_embedding, self.embedding_model.encode(doc, convert_to_tensor=True))))
                for doc in retrieved_documents
            ]
        else:  # Assume OpenAI Embeddings API
            query_embedding = self.embedding_model.embed_query(query)
            scored_documents = [
                (doc, np.dot(query_embedding, self.embedding_model.embed_documents([doc])[0]))
                for doc in retrieved_documents
            ]
        
        return sorted(scored_documents, key=lambda x: x[1], reverse=True)

    def compute_response_quality_score(self, response: str, ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Compute quality score for the generated response using ROUGE, BLEU, and confidence.
        
        Args:
            response (str): Model-generated response.
            ground_truth (str, optional): Ground truth answer if available.
        
        Returns:
            dict: Scores based on ROUGE, BLEU, and a custom confidence metric.
        """
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
        """A custom confidence score based on response characteristics.
        
        Args:
            response (str): Response text to evaluate.
        
        Returns:
            float: Confidence score based on length and specificity.
        """
        length_factor = min(len(response.split()) / 100, 1.0)
        specificity = len(set(response.split())) / len(response.split()) if response else 0
        return (length_factor + specificity) / 2

    def visualize_query_document_similarity(self, query: str, scored_documents: List[Tuple[str, float]]):
        """Visualize query-document similarity as a graph.
        
        Args:
            query (str): Input query text.
            scored_documents (List[Tuple[str, float]]): Documents with their relevance scores.
        """
        G = nx.Graph()
        G.add_node("Query", color="blue", size=300)

        # Add documents as nodes with weighted edges based on similarity
        for idx, (doc, score) in enumerate(scored_documents):
            doc_label = f"Doc {idx+1}"
            G.add_node(doc_label, color="red", size=100)
            G.add_edge("Query", doc_label, weight=score)
        
        # Draw the graph with weighted edges
        pos = nx.spring_layout(G)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, width=edge_weights)
        plt.title("Query-Document Similarity Graph")
        plt.show()
