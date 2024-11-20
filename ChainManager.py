from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

class ChainManager:
    def __init__(self, datastore, model, template):
        """Initialize the ChainManager with datastore, model, and template."""
        self.datastore = datastore
        self.model = model
        self.template = template

    def setup_chain(self):
        """Set up and return the chain for question answering."""
        try:
            parser = StrOutputParser()
            prompt = PromptTemplate.from_template(self.template)
            retriever = self.datastore.as_retriever()

            chain = (
                {
                    "context": itemgetter("question") | retriever,
                    "question": itemgetter("question"),
                }
                | prompt
                | self.model
                | parser
            )
            return chain
        except Exception as e:
            raise RuntimeError(f"Error setting up chain: {e}")
