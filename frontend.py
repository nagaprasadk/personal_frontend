import os
import uuid
import json
from typing import List, Tuple
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
from PyPDF2 import PdfReader
from openai import AzureOpenAI

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# Azure OpenAI Chat Model Config
AZURE_OPENAI_API_KEY_CHAT = os.getenv("AZURE_OPENAI_API_KEY_CHAT")
AZURE_OPENAI_ENDPOINT_CHAT = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")
AZURE_OPENAI_API_VERSION_CHAT = os.getenv("AZURE_OPENAI_API_VERSION_CHAT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Azure OpenAI Embedding Model Config
AZURE_OPENAI_API_KEY_EMBED = os.getenv("AZURE_OPENAI_API_KEY_EMBED")
AZURE_OPENAI_ENDPOINT_EMBED = os.getenv("AZURE_OPENAI_ENDPOINT_EMBED")
AZURE_OPENAI_API_VERSION_EMBED = os.getenv("AZURE_OPENAI_API_VERSION_EMBED")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

# Cosmos DB Config
COSMOS_URI = "https://my-cosmosdb-rag001.documents.azure.com:443/"
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DB = os.getenv("COSMOS_DATABASE", "ragstore")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "vectors")

# Basic checks
missing_vars = []
if not AZURE_OPENAI_API_KEY_CHAT: missing_vars.append("AZURE_OPENAI_API_KEY_CHAT")
if not AZURE_OPENAI_API_KEY_EMBED: missing_vars.append("AZURE_OPENAI_API_KEY_EMBED")
if not COSMOS_KEY: missing_vars.append("COSMOS_KEY")
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}")
    st.stop()

# ----------------------------
# Initialize Azure OpenAI Clients
# ----------------------------
chat_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY_CHAT,
    api_version=AZURE_OPENAI_API_VERSION_CHAT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT_CHAT
)

embed_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY_EMBED,
    api_version=AZURE_OPENAI_API_VERSION_EMBED,
    azure_endpoint=AZURE_OPENAI_ENDPOINT_EMBED
)

# ----------------------------
# Cosmos Client initialization
# ----------------------------
def make_cosmos_client(uri: str, key: str) -> CosmosClient:
    try:
        client = CosmosClient(uri, credential=key)
        _ = list(client.list_databases())[:1]
        return client
    except TypeError:
        client = CosmosClient(uri, key)
        _ = list(client.list_databases())[:1]
        return client

try:
    cosmos_client = make_cosmos_client(COSMOS_URI, COSMOS_KEY)
except Exception as e:
    st.error(f"Failed to connect to Cosmos DB: {e}")
    st.stop()

# Get database & container clients (create if not exists)
try:
    db = cosmos_client.create_database_if_not_exists(id=COSMOS_DB)
    container = db.create_container_if_not_exists(
        id=COSMOS_CONTAINER,
        partition_key="/id",
        offer_throughput=400
    )
except cosmos_exceptions.CosmosHttpResponseError as e:
    st.error(f"Error accessing/creating DB or container: {e}")
    st.stop()

# ----------------------------
# Helper functions
# ----------------------------
def pdf_to_text(file_bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(file_bytes)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def embed_text(text: str) -> List[float]:
    response = embed_client.embeddings.create(
        model=AZURE_OPENAI_EMBED_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def upsert_chunk_to_cosmos(text: str, emb: List[float], metadata: dict):
    item = {
        "id": str(uuid.uuid4()),
        "text": text,
        "embedding": emb,
        "metadata": metadata
    }
    container.upsert_item(item)

def ingest_pdf(file, source_name: str = None):
    pages = pdf_to_text(file)
    total_chunks = 0
    for page_no, page_text in pages:
        chunks = chunk_text(page_text, max_chars=1000, overlap=200)
        for i, ch in enumerate(chunks):
            try:
                emb = embed_text(ch)
            except Exception as e:
                st.error(f"Embedding error: {e}")
                return total_chunks
            meta = {
                "source": source_name or getattr(file, "name", "uploaded_pdf"),
                "page": page_no,
                "chunk_index": i
            }
            upsert_chunk_to_cosmos(ch, emb, meta)
            total_chunks += 1
    return total_chunks

def ingest_text_direct(text: str, source_name: str = "manual"):
    chunks = chunk_text(text, max_chars=1000, overlap=200)
    count = 0
    for i, ch in enumerate(chunks):
        emb = embed_text(ch)
        meta = {"source": source_name, "chunk_index": i}
        upsert_chunk_to_cosmos(ch, emb, meta)
        count += 1
    return count

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

def retrieve_similar(query: str, top_k: int = 5):
    q_emb = embed_text(query)
    items = list(container.read_all_items())
    scored = []
    for it in items:
        emb = it.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(q_emb, emb)
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def answer_with_context(query: str, top_k: int = 5):
    top = retrieve_similar(query, top_k=top_k)
    if not top:
        return "No relevant documents found.", []
    context_texts = []
    sources = []
    for score, it in top:
        context_texts.append(it["text"])
        src = it.get("metadata", {})
        sources.append({"score": float(score), "metadata": src})
    context = "\n\n---\n\n".join(context_texts)
    prompt = (
        "You are a helpful assistant. Use the context provided to answer the question. "
        "If the answer is not contained in the context, say you don't know and do not hallucinate.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    try:
        resp = chat_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers using provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        st.error(f"Azure OpenAI API error: {e}")
        answer = "Error fetching answer from Azure OpenAI."
    return answer, sources

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="PDF → Embeddings → Cosmos RAG", layout="wide")
st.title("📚 RAG: Upload PDF → Embed → Store in Cosmos → Ask")

tab1, tab2 = st.tabs(["Ingest (PDF/Text)", "Query"])

with tab1:
    st.header("Upload PDF to ingest")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    source_name = st.text_input("Source name (optional)", value="")
    if st.button("Ingest PDF"):
        if not uploaded_file:
            st.warning("Upload a PDF first.")
        else:
            with st.spinner("Extracting, embedding and storing... this can take some time"):
                bytes_io = uploaded_file
                count = ingest_pdf(bytes_io, source_name or uploaded_file.name)
            st.success(f"Stored {count} chunks from PDF into Cosmos DB.")

    st.markdown("---")
    st.subheader("Or paste raw text to ingest")
    raw_text = st.text_area("Paste text here", height=200)
    src = st.text_input("Source name for pasted text (optional)", value="")
    if st.button("Ingest Text"):
        if not raw_text.strip():
            st.warning("Enter some text.")
        else:
            with st.spinner("Embedding and storing..."):
                count = ingest_text_direct(raw_text.strip(), source_name=src or "manual")
            st.success(f"Stored {count} chunks into Cosmos DB.")

with tab2:
    st.header("Ask a question")
    q = st.text_input("Enter your question here")
    top_k = st.slider("Number of retrieved chunks (top_k)", min_value=1, max_value=10, value=5)
    if st.button("Search & Answer"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and calling Azure OpenAI..."):
                answer, sources = answer_with_context(q.strip(), top_k=top_k)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Sources (top matches)")
            for s in sources:
                st.markdown(f"- score **{s['score']:.4f}**, metadata: {json.dumps(s['metadata'])}")
