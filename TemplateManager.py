from Templates import TEMPLATES

class TemplateManager:
    def __init__(self):
        """Initialize the TemplateManager."""
        self.templates = TEMPLATES

    def get_template(self, template_name: str = "default") -> str:
        """Retrieve a specific template by name."""
        print(f"Retrieving template: {template_name}")
        return self.templates.get(template_name, self.templates["default"])