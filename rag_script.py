import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

# Load environment variables
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

# 1. Configuration: Local Embeddings & Gemini LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=google_api_key) # Adjusted model name

# 2. Setup Data and Vector Store
print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("portfolio_index")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Build/Load Index
print("Initializing index...")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# 4. Interactive Chat Loop
query_engine = index.as_query_engine()

print("\n" + "="*30)
print("PORTFOLIO CHATBOT READY")
print("Type 'exit' to stop.")
print("="*30)

while True:
    user_input = input("\nAsk about the portfolio: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    if user_input.strip():
        print("🔍 Searching...")
        response = query_engine.query(user_input)
        print(f"\nAI Response: {response}")
    else:
        print("Please enter a question.")