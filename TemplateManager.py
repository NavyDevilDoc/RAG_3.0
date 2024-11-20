import json
from typing import Dict

class TemplateManager:
    def __init__(self, template_path: str = None):
        """Initialize the TemplateManager.

        Args:
            template_path (str): Path to a JSON file containing predefined templates.
                                 If None, a default template is used.
        """
        self.template_path = template_path
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load templates from a JSON file or set a default template.

        Returns:
            dict: Dictionary of templates with keys as template names.
        """
        if self.template_path:
            try:
                with open(self.template_path, "r") as file:
                    templates = json.load(file)
                print(f"Templates loaded from {self.template_path}")
                return templates
            except Exception as e:
                print(f"Error loading templates: {e}")
                return {"default": self._default_template()}
        else:
            print("Using default template.")
            return {"default": self._default_template()}

    '''
    @staticmethod
    def _default_template() -> str:
        """Provide a default template."""
        return """
        Based on the context provided, answer the question with a detailed and well-supported explanation. 
        Rely exclusively on information from the designated documentation.
        Avoid providing extraneous information.

        If the question is unclear or lacks sufficient context for a confident answer, 
        respond with "I don't know" or ask for additional clarification.

        Spell out all acronyms upon first mention.
        Reference specific sections or points from the context to substantiate major points.

        Context: {context}
        Question: {question}
        """
    '''
    def get_template(self, template_name: str = "default") -> str:
        """Retrieve a specific template by name.

        Args:
            template_name (str): Name of the template to retrieve. Defaults to "default".

        Returns:
            str: The template string.
        """
        if template_name in self.templates:
            return self.templates[template_name]
        else:
            print(f"Template '{template_name}' not found. Using default template.")
            return self.templates["default"]

    def add_template(self, template_name: str, template_content: str):
        """Add a new template to the collection.

        Args:
            template_name (str): Name of the new template.
            template_content (str): The template content.
        """
        self.templates[template_name] = template_content
        print(f"Template '{template_name}' added successfully.")

    def save_templates(self, save_path: str):
        """Save the current templates to a JSON file.

        Args:
            save_path (str): Path to save the templates file.
        """
        try:
            with open(save_path, "w") as file:
                json.dump(self.templates, file, indent=4)
            print(f"Templates saved to {save_path}")
        except Exception as e:
            print(f"Error saving templates: {e}")
