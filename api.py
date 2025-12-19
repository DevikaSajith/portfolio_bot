import os
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from fastapi.middleware.cors import CORSMiddleware

# 1. Setup & Configuration
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

app = FastAPI()

# Enable CORS so your website can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your actual domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=google_api_key)

# 2. Load the Index (Optimized for API)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("portfolio_index")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load existing index without re-reading documents for speed
index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

SYSTEM_PROMPT = """
You are the personal AI Assistant for Devika P Sajith. 
Your goal is to answer questions about Devika's skills, projects, and certificates accurately.
- Introduce yourself as 'Devika's Personal Assistant' if asked who you are.
- Devika is a Co-Founder of Ayunex and an R&D Intern at Dhee Yantra.
- If a user asks for a certificate, provide the corresponding '[Click to View Certificate]' link.
- Be professional, encouraging, and concise.
"""

query_engine = index.as_query_engine(system_prompt=SYSTEM_PROMPT)

# 3. API Models & Endpoints
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = query_engine.query(request.message)
        return {"answer": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "Devika's Portfolio API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)