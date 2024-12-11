# ConversationHistory.py
from Driver import Driver
from TextPreprocessor import TextPreprocessor

class ConversationHistory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_history(self):
        return self.messages


# Modify the driver class to include conversation history
def process_query(self, input_text, conversation_history=None):
    if conversation_history is None:
        conversation_history = ConversationHistory()
    
    # Add user's message to history
    conversation_history.add_message("user", input_text)
    
    # Include conversation history in the prompt
    full_context = "\n".join([f"{m['role']}: {m['content']}" 
                             for m in conversation_history.get_history()])
    
    # Get LLM response with context
    response = self.llm.generate(full_context)
    
    # Add assistant's response to history
    conversation_history.add_message("assistant", response)
    
    return response, conversation_history


# Modify the main LLM mode section:
driver = Driver()
if driver.mode == 'llm':
    text_preprocessor = TextPreprocessor()
    conversation_history = ConversationHistory()
    
    print("Enter your questions (type 'exit' to end the conversation):")
    while True:
        user_input = input("\nQ: ").strip()
        if user_input.lower() == 'exit':
            break
            
        response, conversation_history = driver.process_query(
            user_input, 
            conversation_history
        )
        print(f"A: {response}")