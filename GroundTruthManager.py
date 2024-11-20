import json
from typing import Dict

class GroundTruthManager:
    def __init__(self, ground_truth_path: str = None):
        """Initialize the GroundTruthManager.

        Args:
            ground_truth_path (str): Path to a JSON file containing ground truth data.
                                     If None, an empty ground truth dictionary is initialized.
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, str]:
        """Load ground truth data from a JSON file or initialize an empty dictionary.

        Returns:
            dict: Ground truth data mapping questions to answers.
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
        """Retrieve the ground truth answer for a given question.

        Args:
            question (str): The question to retrieve the answer for.

        Returns:
            str: The ground truth answer or a default message if the question is not found.
        """
        return self.ground_truth.get(question, "No ground truth available for this question.")

    def add_ground_truth(self, question: str, answer: str):
        """Add a new question-answer pair to the ground truth.

        Args:
            question (str): The question to add.
            answer (str): The corresponding answer to add.
        """
        self.ground_truth[question] = answer
        print(f"Added ground truth for question: {question}")

    def save_ground_truth(self, save_path: str = None):
        """Save the current ground truth data to a JSON file.

        Args:
            save_path (str): Path to save the ground truth data. Defaults to the original file path.
        """
        path = save_path or self.ground_truth_path
        if not path:
            print("No save path provided. Unable to save ground truth.")
            return

        try:
            with open(path, "w") as file:
                json.dump(self.ground_truth, file, indent=4)
            print(f"Ground truth saved to {path}")
        except Exception as e:
            print(f"Error saving ground truth: {e}")
