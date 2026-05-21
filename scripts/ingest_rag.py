"""
Ingest all JSON data files into a ChromaDB vector store.

Run from the project root:
    python scripts/ingest_rag.py
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
JSON_DIR = BASE_DIR / "data" / "json"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"
COLLECTION_NAME = "sgtech_ai_navigator"


# ---------------------------------------------------------------------------
# Text chunk builders
# ---------------------------------------------------------------------------

def _j(value) -> str:
    """Join list to string, or stringify scalar."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return str(value) if value is not None else ""


def build_text_chunk(record: dict, source_type: str) -> str:
    j = _j
    if source_type == "tools":
        return (
            f"Tool: {j(record.get('tool_name'))} by {j(record.get('vendor'))}. "
            f"Category: {j(record.get('category'))}. "
            f"Business functions: {j(record.get('business_functions'))}. "
            f"Use cases: {j(record.get('use_cases'))}. "
            f"Description: {j(record.get('description'))}. "
            f"SME suitability: {j(record.get('sme_suitability'))}. "
            f"Pricing: {j(record.get('pricing_tier'))}. "
            f"Implementation complexity: {j(record.get('implementation_complexity'))}. "
            f"Risk level: {j(record.get('risk_level'))}. "
            f"Governance tags: {j(record.get('governance_tags'))}."
        )
    if source_type == "grants":
        return (
            f"Grant: {j(record.get('grant_name'))} ({j(record.get('short_name'))}). "
            f"Description: {j(record.get('description'))}. "
            f"Supported categories: {j(record.get('supported_categories'))}. "
            f"Business functions: {j(record.get('business_functions'))}. "
            f"Eligibility: {j(record.get('eligibility_hints'))}. "
            f"Estimated support: {j(record.get('estimated_support'))}. "
            f"Keywords: {j(record.get('keywords'))}."
        )
    if source_type == "use_cases":
        return (
            f"Use case: {j(record.get('title'))}. "
            f"Business function: {j(record.get('business_function'))}. "
            f"Pain points: {j(record.get('common_pain_points'))}. "
            f"Recommended tool categories: {j(record.get('recommended_tool_categories'))}. "
            f"Recommended grants: {j(record.get('recommended_grants'))}. "
            f"Sample user prompt: {j(record.get('sample_user_prompt'))}. "
            f"Expected output: {j(record.get('expected_output_type'))}."
        )
    if source_type == "starter_kits":
        return (
            f"Starter Kit: {j(record.get('title'))}. "
            f"Business function: {j(record.get('business_function'))}. "
            f"Objective: {j(record.get('objective'))}. "
            f"Workflow steps: {j(record.get('workflow_summary'))}. "
            f"Required data: {j(record.get('required_data'))}. "
            f"KPIs to track: {j(record.get('suggested_kpis'))}. "
            f"Implementation steps: {j(record.get('implementation_steps'))}. "
            f"Human review points: {j(record.get('human_review_points'))}."
        )
    if source_type == "sgtech_events":
        return (
            f"SGTech Event: {j(record.get('event_name'))}. "
            f"Category: {j(record.get('event_category'))}. "
            f"Target audience: {j(record.get('target_audience'))}. "
            f"Description: {j(record.get('description'))}. "
            f"Member price: SGD {j(record.get('member_price'))}. "
            f"Non-member price: SGD {j(record.get('non_member_price'))}."
        )
    if source_type == "membership_details":
        return (
            f"SGTech Membership: {j(record.get('membership_type'))}. "
            f"Annual revenue band: {j(record.get('annual_revenue_band'))}. "
            f"Annual fee: SGD {j(record.get('annual_fee_sgd'))}. "
            f"Registration fee: SGD {j(record.get('registration_fee_sgd'))}. "
            f"Benefits: {j(record.get('benefits_summary'))}. "
            f"Included chapters: {j(record.get('included_chapters'))}. "
            f"Additional chapter fee: SGD {j(record.get('additional_chapter_fee'))}."
        )
    # Fallback: dump entire record as JSON text
    return json.dumps(record)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(record: dict, source_type: str) -> dict:
    meta: dict = {"source_type": source_type}

    # Scalar fields that exist in one or more source types
    scalar_fields = [
        "business_function",
        "category",
        "risk_level",
        "membership_type",
    ]
    for field in scalar_fields:
        val = record.get(field)
        if val is not None:
            meta[field] = _j(val)

    # List fields — flatten to comma-separated string for ChromaDB
    list_fields = ["business_functions", "governance_tags"]
    for field in list_fields:
        val = record.get(field)
        if val is not None:
            meta[field] = _j(val)

    # ID fields
    for id_field in ["tool_id", "grant_id", "use_case_id", "starter_kit_id", "event_id"]:
        val = record.get(id_field)
        if val is not None:
            meta[id_field] = str(val)

    return meta


def get_doc_id(record: dict, source_type: str, idx: int) -> str:
    id_field_map = {
        "tools": "tool_id",
        "grants": "grant_id",
        "use_cases": "use_case_id",
        "starter_kits": "starter_kit_id",
        "sgtech_events": "event_id",
    }
    id_field = id_field_map.get(source_type)
    if id_field:
        val = record.get(id_field)
        if val:
            return f"{source_type}_{val}"
    if source_type == "membership_details":
        slug = _j(record.get("membership_type", f"item_{idx}"))
        return f"membership_{slug.replace(' ', '_').replace('/', '_')}"
    return f"{source_type}_{idx}"


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Create a .env file with it set.")

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # Delete existing collection for a clean re-ingest
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for json_file in sorted(JSON_DIR.glob("*.json")):
        source_type = json_file.stem
        records: list[dict] = json.loads(json_file.read_text(encoding="utf-8"))

        documents, metadatas, ids = [], [], []
        for idx, record in enumerate(records):
            ids.append(get_doc_id(record, source_type, idx))
            documents.append(build_text_chunk(record, source_type))
            metadatas.append(extract_metadata(record, source_type))

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"  Ingested {len(records):>3} records  <-  {json_file.name}")
        total += len(records)

    print(f"\nDone. {total} records stored in collection '{COLLECTION_NAME}'")
    print(f"Vector DB path: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
