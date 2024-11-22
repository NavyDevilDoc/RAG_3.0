# GroundTruthManager.py

"""
GroundTruthManager.py

A module for managing ground truth data for question-answering evaluation.

Features:
- JSON-based ground truth storage
- Question-answer pair management
- Error handling and validation 
- Answer retrieval and updates
"""

import json
from typing import Dict, Optional

class GroundTruthManager:
    """
    Manages ground truth data for evaluating question-answering systems.

    Features:
    1. Load ground truth from JSON
    2. Add/update question-answer pairs
    3. Retrieve ground truth answers
    4. Save updates to storage

    Attributes:
        ground_truth_path (Optional[str]): Path to ground truth JSON file
        ground_truth (Dict[str, str]): Dictionary of question-answer pairs

    Example:
        >>> manager = GroundTruthManager("ground_truth.json")
        >>> answer = manager.get_answer("What is RAG?")
    """
    def __init__(self, ground_truth_path: str = None):
        """
        Initialize ground truth manager.

        Args:
            ground_truth_path (Optional[str]): Path to ground truth JSON file

        Raises:
            ValueError: If file path is invalid
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, str]:
        """
        Load ground truth data from JSON file.

        Returns:
            Dict[str, str]: Ground truth question-answer pairs

        Raises:
            FileNotFoundError: If ground truth file not found
            json.JSONDecodeError: If JSON parsing fails
        """
        if self.ground_truth_path:
            try:
                with open(self.ground_truth_path, "r") as file:
                    data = json.load(file)
                print(f"Ground truth loaded from {self.ground_truth_path}")
                return data
            except Exception as e:
                print(f"Error loading ground truth: {e}")
                return {}
        else:
            print("No ground truth file provided. Initializing empty ground truth.")
            return {}

    def get_answer(self, question: str) -> str:
        """
        Get ground truth answer for question.

        Args:
            question (str): Question to look up

        Returns:
            str: Ground truth answer or default message

        Example:
            >>> answer = manager.get_answer("What is RAG?")
            >>> print(answer)
        """
        return self.ground_truth.get(question, "No ground truth available for this question.")

    def add_ground_truth(self, question: str, answer: str):
        """
        Add or update ground truth pair.

        Args:
            question (str): Question to add/update
            answer (str): Ground truth answer

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If save fails

        Example:
            >>> manager.add_ground_truth(
            ...     "What is RAG?",
            ...     "Retrieval Augmented Generation..."
            ... )
        """
        self.ground_truth[question] = answer
        print(f"Added ground truth for question: {question}")

    def save_ground_truth(self, save_path: Optional[str] = None) -> None:
        """
        Save the current ground truth data to a JSON file.

        Args:
            save_path (Optional[str]): Path to save ground truth data. 
                Defaults to original file path if None.

        Raises:
            ValueError: If no valid save path available
            IOError: If file write fails
            json.JSONEncodeError: If JSON serialization fails

        Example:
            >>> manager.add_ground_truth("New Q", "New A")
            >>> manager.save_ground_truth("updated_truth.json")
            Ground truth saved to updated_truth.json
        """
        path = save_path or self.ground_truth_path
        if not path:
            raise ValueError("No save path provided. Unable to save ground truth.")

        try:
            with open(path, "w") as file:
                json.dump(self.ground_truth, file, indent=4)
            print(f"Ground truth saved to {path}")
        except Exception as e:
            error_msg = f"Error saving ground truth: {e}"
            print(error_msg)
            raise IOError(error_msg)