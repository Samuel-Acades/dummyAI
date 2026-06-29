import os
import sys
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from google import genai
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
from docx import Document

from app.config import get_runtime_settings

warnings.filterwarnings("ignore", message="ARC4 has been moved to cryptography.hazmat.decrepit")

app = FastAPI(title="🌱 Kobiri AI Production Backend", version="1.0.0")
settings = get_runtime_settings()

# 1. Initialize Global API Gateways
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY environment variable is missing!")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# 2. Spin up Local AI Architecture
print("📥 Initializing Local Semantic Embedding Engine...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("🗄️ Initializing Local Vector Database (ChromaDB)...")
# Stores data persistently in a local folder named 'chroma_db'
chroma_client = chromadb.PersistentClient(path=settings["chroma_db_path"])
collection = chroma_client.get_or_create_collection(name="kobiri_agricultural_policy")

# 3. Data Ingestion & Extraction Engine
DATA_DIR = settings["data_dir"]

def populate_vector_db():
    if collection.count() > 0:
        print(f"✅ Vector index already populated with {collection.count()} blocks. Skipping parsing.")
        return

    print("📖 Scanning data directory for raw sources...")
    chunks = []
    
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Warning: Data directory '{DATA_DIR}' not found.")
        return

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        
        if ext == ".pdf":
            try:
                reader = PdfReader(file_path)
                text = "".join([page.extract_text() or "" for page in reader.pages])
                # INCREASE chunk size to 1500 characters, step by 1000 (creates fewer, richer blocks)
                for i in range(0, len(text), 1000):
                    c = text[i:i+1500].strip()
                    if len(c) > 100: 
                        chunks.append((f"{file_name}_chunk_{i}", c, {"source": file_name}))
            except Exception as e:
                print(f"Error reading PDF {file_name}: {str(e)}")
                
        elif ext == ".docx":
            try:
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                for i in range(0, len(text), 1000):
                    c = text[i:i+1500].strip()
                    if len(c) > 100: 
                        chunks.append((f"{file_name}_chunk_{i}", c, {"source": file_name}))
            except Exception as e:
                print(f"Error reading Docx {file_name}: {str(e)}")
                
        elif ext == ".jsonl":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if line.strip():
                            chunks.append((f"{file_name}_line_{idx}", line.strip(), {"source": file_name}))
            except Exception as e:
                print(f"Error reading JSONL {file_name}: {str(e)}")

    if chunks:
        print(f"🧬 Vectorizing and uploading {len(chunks)} structural context blocks into ChromaDB...")
        
        # Process in chunks of 100 to save RAM and avoid timeouts
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            ids = [item[0] for item in batch]
            documents = [item[1] for item in batch]
            metadatas = [item[2] for item in batch]
            
            # Generate mathematical embeddings on CPU
            embeddings = embedding_model.encode(documents, show_progress_bar=False).tolist()
            
            # Insert batch into database
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            print(f"📦 Indexed batch {i // batch_size + 1} / {(len(chunks) + batch_size - 1) // batch_size}")
            
        print("✅ Vector database ingestion run completed successfully!")

@app.on_event("startup")
def startup_event():
    populate_vector_db()

# 4. Pydantic Schemes for Data Validation
class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str

class ChatPayload(BaseModel):
    message: str
    history: List[ChatMessage] = []

# 5. Core API Endpoint
@app.post("/api/v1/chat")
async def handle_kobiri_chat(payload: ChatPayload):
    try:
        # Step A: Convert incoming query to embedding vector
        query_vector = embedding_model.encode(payload.message).tolist()
        
        # Step B: Instantly query local ChromaDB vector store
        results = collection.query(query_embeddings=[query_vector], n_results=15)
        
        # Combine text elements
        retrieved_documents = results.get("documents", [[]])[0]
        relevant_context = "\n\n".join(retrieved_documents) if retrieved_documents else "No direct matching policy references found."

        # Step C: Format production dynamic instructions
        dynamic_guidance = (
                        """
            You are Kobiri AI, an expert AI assistant specializing in the Malawi National Agricultural and Fertilizer Policies.

            CRITICAL LANGUAGE RULES:
            1. Detect the language used by the user in their query.
            2. If the user asks a question in Chichewa, you MUST reply entirely in clear, natural Chichewa (e.g., if they ask about 'fetereza', explain the policy guidelines in Chichewa).
            3. If the user asks a question in English, you MUST reply entirely in English.
            4. Always ground your responses strictly in the provided structural context from the policy documents. If the context does not contain the answer, politely state that you do not know in the matching language.
            """
            "following semantically matched technical document context:\n\n"
            f"=== START DOCUMENT CONTEXT ===\n{relevant_context}\n=== END DOCUMENT CONTEXT ===\n\n"
            "Rules:\n"
            "1. Only use facts directly mentioned in the document context above.\n"
            "2. If the context doesn't contain the answer, politely tell the user what parts of agricultural policy you can discuss.\n"
            "3. Respond naturally and politely in the language used by the user (English or Chichewa)."
        )

        # Step D: Structure multi-turn conversational payloads
        formatted_contents = []
        for turn in payload.history:
            formatted_contents.append({
                "role": "user" if turn.role == "user" else "model",
                "parts": [{"text": turn.content}]
            })
            
        # Append latest entry
        formatted_contents.append({"role": "user", "parts": [{"text": payload.message}]})

        # Step E: Query Cloud Model Engine
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=formatted_contents,
            config={
                'system_instruction': dynamic_guidance,
                'temperature': 0.1,
            }
        )
        
        if response.text:
            return {"status": "success", "response": response.text}
        raise HTTPException(status_code=500, detail="Model returned an empty string validation response.")
        
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Internal Server Pipeline Error: {str(err)}")