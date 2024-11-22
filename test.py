from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from groq import Groq  # Correct the import statement for Groq
import os

import warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        
        # Setup Groq Client
        self.groq_api_key = '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'
        self.groq_client = Groq(api_key=self.groq_api_key)  # Use the Groq client directly

        # Set up global settings
        Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self._index = None
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        print(memory)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"D:\VoiceAssistant\mydata.txt"]
            )
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="store_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def fetch_groq_data(self, groq_query):
        response = self.groq_client.search(groq_query)  # or some other method
        if response.status_code == 200:
           data = response.json()
           return data
        else:
            print(f"Error: {response.status_code} — {response.text}")
            return None


    def interact_with_llm(self, customer_query):
        # Example: Fetching product data based on the query
        groq_query = """
        *[_type == "product"] | order(name asc) {
            name,
            description,
            category,
            price
        }
        """
        products = self.fetch_groq_data(groq_query)

        if products:
            product_list = ", ".join([f"{product['name']} for {product['price']}" for product in products])
            self._prompt = f"**Instructions:**\n" \
                           f"Respond with short and engaging answers. You can suggest products like: {product_list}\n" \
                           f"{self._prompt}"

        AgentChatResponse = self._chat_engine.chat(customer_query)
        answer = AgentChatResponse.response
        return answer

    @property
    def _prompt(self):
        return """
    **Instructions:**
    Respond only with short and engaging answers. Keep interactions brief and to the point.

    **Beginning the Conversation**
    "Hello! Welcome to Fresh Mart! What's your name?"

    (User provides name)
    "Nice to meet you, [User Name]! Last time, you liked [popular items]. Want any today?"

    **Throughout the Conversation**
    - "Need sugar or flour?"
    - "Fresh veggies just arrived! Interested?"
    - "How about snacks? I can suggest [items]."
    - "Your favorites like [favorite items]? Add?"

    (User adds items)
    - "Got it! Added [items]. Anything else?"

    (User declines or adds more)
    - "No worries! Have you seen our new arrivals?"

    (If user shows interest in unhealthy items)
    - If sugar: "Try jaggery! It's healthier!"
    - For unhealthy items: "How about [healthy alternative]?"

    **Cooking Assistance**
    - If user mentions a dish:
      - "For curry, you need onions and spices. Add?"
      - "Baking? Grab flour and jaggery?"
      - "We have paneer for palak paneer. Want some?"

    **Confirming and Summarizing the Order**
    "You’ve picked [items]. Anything else?"

    **Calculating the Total Amount**
    - If user asks for a total:
      - "Your total is [total amount]."

    **Ending the Conversation**
    "Your order is [full order] for Rupees [total amount]. Thanks, [User Name]! Have a great day!"

    **Note:**
    Only answer shopping-related queries. For anything else, say:
    "I'm here for shopping help! Ask me about products."
    """
        
# Run the assistant
def main():
    ai_assistant = AIVoiceAssistant()
    response = ai_assistant.interact_with_llm("Hello, what fresh items do you have?")
    print(response)

# Run the main function
if __name__ == "__main__":
    main()
