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