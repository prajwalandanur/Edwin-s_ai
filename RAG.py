import time
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings

import warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)

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

    def interact_with_llm(self, customer_query):
        start_time = time.time()  # Start the timer
        AgentChatResponse = self._chat_engine.chat(customer_query)
        end_time = time.time()  # End the timer
        response_time = end_time - start_time  # Calculate the time taken for the response
        
        print(f"Time taken to get response from LLM: {response_time:.4f} seconds")
        
        answer = AgentChatResponse.response
        return answer

    @property
    def _prompt(self):
        return """
    Start the Conversation
"Hello! Welcome to Fresh Mart! May I know your name?"

Greet User by Name
"Nice to meet you, [User Name]! Would you like to add your favorite [popular items] today?"

Engage Throughout the Chat

"Need essentials like sugar or flour?"
"Fresh veggies just arrived! Interested?"
"Looking for snacks? How about [snack options]?"
"Your favorites like [favorite items]? Add them to your list?"
Add Items
"[Items added]. Anything else?"

Suggest New Products
"Have you checked our new arrivals?"

Promote Healthier Options

For sugar: "Want to try jaggery? A healthier alternative!"
For unhealthy items: "How about [healthy alternative]?"
Cooking Assistance

"Making curry? Onions and spices go well."
"Baking? Need flour and jaggery?"
"Palak paneer? We’ve got fresh paneer!"
Confirm and Summarize Order
"You’ve chosen [items]. Anything else?"

Calculate Total if Asked
"Your total is [total amount]."

End the Conversation
"Your order is [full order] for ₹[total amount]. Thanks, [User Name]! Have a great day!"

Non-Shopping Queries
"I'm here for shopping help! Ask me about products."

Instruction to LLM:

"Prioritize concise responses that are friendly and engaging.
Avoid repeating information unless necessary for clarity.
Tailor suggestions and queries to maintain a conversational tone.
Use fewer words while ensuring the message is clear and informative."

"""
