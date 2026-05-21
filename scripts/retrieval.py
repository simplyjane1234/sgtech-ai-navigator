"""
Search the ChromaDB knowledge base.

As a module:
    from scripts.retrieval import search_knowledge_base
    results = search_knowledge_base("automate lead generation", top_k=5)

As a CLI:
    python scripts/retrieval.py "I want to automate lead generation"
    python scripts/retrieval.py "What grants help with CRM?" 3
"""

import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
COLLECTION_NAME = "sgtech_ai_navigator"


def _get_collection():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Create a .env file with it set.")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)


def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the knowledge base for chunks relevant to query.

    Returns a list of dicts, each with:
        text      - the matched document chunk
        metadata  - source_type, ids, and any extracted fields
        score     - cosine similarity in [0, 1]; higher is more relevant
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "metadata": meta,
            "score": round(1 - dist, 4),  # cosine distance -> similarity
        })
    return hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("=" * 70)
    for i, hit in enumerate(results, 1):
        meta = hit["metadata"]
        source = meta.get("source_type", "?")
        score = hit["score"]
        # Pick the most descriptive ID available
        record_id = (
            meta.get("tool_id")
            or meta.get("grant_id")
            or meta.get("use_case_id")
            or meta.get("starter_kit_id")
            or meta.get("event_id")
            or meta.get("membership_type")
            or "—"
        )
        print(f"\n[{i}] Score: {score:.4f}  |  Source: {source}  |  ID: {record_id}")
        print(f"    {hit['text'][:300]}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/retrieval.py \"your query\" [top_k]")
        sys.exit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    _print_results(query, search_knowledge_base(query, top_k=top_k))
