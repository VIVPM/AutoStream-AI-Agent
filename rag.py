import json
import os
import chromadb
from google import genai


KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Singletons
_chroma_client = None
_genai_client = None


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _genai_client


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using gemini-embedding-001 with 384 dimensions."""
    client = get_genai_client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config={
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 384,
        },
    )
    return [e.values for e in result.embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query for retrieval."""
    client = get_genai_client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
        config={
            "task_type": "RETRIEVAL_QUERY",
            "output_dimensionality": 384,
        },
    )
    return result.embeddings[0].values


def load_knowledge_base() -> tuple[list[str], list[dict]]:
    """Load knowledge base JSON and return documents + metadata."""
    with open(KNOWLEDGE_BASE_PATH, "r") as f:
        data = json.load(f)

    documents = []
    metadatas = []

    # Company overview
    documents.append(f"{data['company']} - {data['tagline']}")
    metadatas.append({"source": "overview"})

    # Plans
    for plan in data["plans"]:
        features = ", ".join(plan["features"])
        content = f"{plan['name']}: {plan['price']}. Features: {features}"
        documents.append(content)
        metadatas.append({"source": "pricing", "plan": plan["name"]})

    # Policies
    for policy in data["policies"]:
        documents.append(policy)
        metadatas.append({"source": "policy"})

    # FAQ
    for faq in data["faq"]:
        content = f"Q: {faq['question']} A: {faq['answer']}"
        documents.append(content)
        metadatas.append({"source": "faq"})

    return documents, metadatas


def build_vector_store():
    """Build or load ChromaDB collection."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="autostream_kb")

    # Only populate if empty
    if collection.count() == 0:
        documents, metadatas = load_knowledge_base()
        embeddings = embed_texts(documents)
        ids = [f"doc_{i}" for i in range(len(documents))]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve relevant context from the knowledge base."""
    collection = build_vector_store()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    # Filter by relevance — only include docs with distance < threshold
    documents = results["documents"][0]
    distances = results["distances"][0] if "distances" in results else None

    if distances:
        filtered = [doc for doc, dist in zip(documents, distances) if dist < 1.2]
        if filtered:
            return "\n".join(filtered)

    return "\n".join(documents)
