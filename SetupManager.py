from ModelManager import ModelManager
from ComputeResourceManager import ComputeResourceManager

class SetupManager:
    def __init__(self, env_path: str):
        self.model_manager = ModelManager(env_path)
        self.resource_manager = ComputeResourceManager().get_compute_settings()

    def load_resources(self, config: dict, llm_index: int, embed_index: int):
        # Validate selections
        self.model_manager.validate_selection(config["selected_llm_type"], 
                                              self.model_manager.llm_choices.keys())
        self.model_manager.validate_selection(config["selected_embedding_scheme"], 
                                              self.model_manager.embedding_choices.keys())

        # Load model and embeddings
        selected_llm = self.get_selected_llm(config, llm_index)
        selected_embed = self.model_manager.embedding_choices[config["selected_embedding_scheme"]][embed_index]
        model = self.model_manager.load_model(config["selected_llm_type"], selected_llm, 
                                              self.resource_manager)
        embeddings = self.model_manager.load_embeddings(config["selected_embedding_scheme"], 
                                                        selected_embed)
        dimensions = self.model_manager.determine_embedding_dimensions(embeddings)
        return model, embeddings, dimensions, selected_embed

    def get_selected_llm(self, config: dict, llm_index: int):
        return self.model_manager.llm_choices[config["selected_llm_type"]][llm_index]
