from ScoringMetric import ScoringMetric
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class ResponseSelector:
    REASONING_MARKERS = ['because', 'therefore', 'however']
    RELEVANCE_WEIGHT = 0.7
    CONFIDENCE_WEIGHT = 0.3

    def __init__(self, model_name: str = "all-mpnet-base-v2", top_results: int = 15, use_reranking: bool = True, save_outputs: bool = False, output_file_path: str = "re-ranking_test_outputs.txt"):
        """Initialize response selector with embedding model and top_k parameter."""
        try:
            self.embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError(f"Error loading embedding model '{model_name}': {e}")
        if top_results <= 0:
            raise ValueError("top_k must be a positive integer.")
        self.top_results = top_results
        self.use_reranking = use_reranking
        self.save_outputs = save_outputs
        self.output_file_path = output_file_path
        self.scoring_metric = ScoringMetric(model_name, 'sentence_transformer')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')


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


    # Normalize the scores since they are generated using different embedding models
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to a range of 0 to 1."""
        min_score = min(scores)
        max_score = max(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]


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

        # Normalize scores before re-ranking
        scores = [score for _, score in scored_responses]
        normalized_scores = self._normalize_scores(scores)
        scored_responses = [(response, score) for (response, _), score in zip(scored_responses, normalized_scores)]

        '''
        # Re-rank documents
        if self.use_reranking:
            original_responses = [response for response, _ in scored_responses]
            reranked_responses = self.rerank_documents(question, original_responses)
            if self.save_outputs:
                self.save_outputs_to_file(original_responses, reranked_responses)
            scored_responses = [(response, score) for response, score in scored_responses if response in reranked_responses]

            # Print re-ranked scores for debugging
            print("-----Similarity scores after re-ranking and normalizing-----")
            for response, score in scored_responses:
                print(f"Response: {response[:30]}... Score: {score}")
        '''
        # Sort and return only the top n responses
        try:
            return sorted(scored_responses, key=lambda x: x[1], reverse=True)[:self.top_results]
        except Exception as e:
            print(f"Error sorting responses: {e}")
            return []


    def rerank_documents(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Re-rank documents using BERT-based similarity."""
        query_embedding = self._encode_text(query)
        doc_embeddings = [self._encode_text(doc) for doc in documents]
        similarities = [self._cosine_similarity(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]

        # Print similarity scores for debugging
        print("-----Similarity scores before re-ranking-----")
        for doc, score in zip(documents, similarities):
            print(f"Document: {doc[:30]}... Score: {score}")

        ranked_docs_with_scores = sorted(zip(similarities, documents), key=lambda x: x[0], reverse=True)
        ranked_docs = [(doc, score) for score, doc in ranked_docs_with_scores]

        '''
        # Print ranked documents with scores for debugging
        print("Documents after re-ranking:")
        for score, doc in ranked_docs_with_scores:
            print(f"Document: {doc[:30]}... Score: {score}")
        '''

        return ranked_docs


    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using BERT."""
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


    def _cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors."""
        return torch.nn.functional.cosine_similarity(tensor1, tensor2).item()


    def save_outputs_to_file(self, original_outputs: List[str], reranked_outputs: List[str]):
        """Save original and re-ranked outputs to a text file."""
        with open(self.output_file_path, "w") as file:
            file.write("Original Outputs:\n")
            for output in original_outputs:
                file.write(output + "\n")
            file.write("\nRe-ranked Outputs:\n")
            for output in reranked_outputs:
                file.write(output + "\n")


    def select_best_response(self, ranked_responses: List[Tuple[str, float]]) -> str:
        """Select single best response from pre-ranked candidates."""
        try:
            return ranked_responses[0][0] if ranked_responses else "No suitable response found."
        except Exception as e:
            print(f"Error selecting best response: {e}")
            return "No suitable response found."
        

    def save_results_to_dataframe(self, original_results, re_ranked_results):
        """Save original and re-ranked results to a DataFrame and save to a CSV file."""
        original_results_len = len(original_results)
        re_ranked_results_len = len(re_ranked_results)

        if original_results_len != re_ranked_results_len:
            print("Warning: The lengths of the original and re-ranked results are not the same.")
        
        data = {
            "Original Result": [result[0] for result in original_results],
            "Rank": list(range(1, original_results_len + 1)),
            "Score": [result[1] for result in original_results],
            "Re-ranked Result": [result[0] for result in re_ranked_results],
            "New Rank": list(range(1, re_ranked_results_len + 1)),
            "New Score": [result[1] for result in re_ranked_results]
        }
        
        # Ensure all lists have the same length
        min_len = min(original_results_len, re_ranked_results_len)
        for key in data:
            data[key] = data[key][:min_len]
        
        df = pd.DataFrame(data)
        
        return df