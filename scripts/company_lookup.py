"""
Look up a Singapore company by name and UEN.

Searches the web for publicly available information, then uses GPT to extract
a structured company profile. Fails gracefully — if nothing is found, returns
a minimal profile built from what the user provided.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

UEN_HELP = "https://www.uen.gov.sg"


def _web_search(query: str, max_results: int = 5) -> list[str]:
    """Return a list of text snippets from a DuckDuckGo search."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [f"{r['title']}: {r['body']}" for r in results if r.get("body")]
    except Exception:
        return []


def _extract_profile(company_name: str, uen: str, snippets: list[str]) -> dict:
    """Ask GPT to extract a structured company profile from search snippets."""
    context = "\n".join(snippets) if snippets else "(No search results found.)"

    prompt = f"""You are helping personalise an AI tool recommender for Singapore SMEs.

A user has entered their company details. Use the search results below to build a brief company profile.

Company name: {company_name}
UEN: {uen}

Search results:
{context}

Return a JSON object with exactly these fields:
- name: company name as provided
- uen: UEN as provided
- industry: main industry or sector (e.g. "F&B", "Retail", "Technology", "Professional Services", "Healthcare", "Logistics")
- description: 1-2 sentences describing what the company does
- size_estimate: one of "Startup", "SME", or "Unknown"
- key_activities: list of up to 3 main business activities (short phrases)

If a field cannot be determined from the search results, use a sensible default or "Unknown".
Do not invent facts — only use what is reasonably supported by the search results or the company name."""

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def lookup_company(company_name: str, uen: str) -> dict:
    """
    Look up a Singapore company by name and UEN.

    Returns a dict with: name, uen, industry, description, size_estimate, key_activities.
    Always returns something — falls back to a minimal profile on failure.
    """
    # Search with both name and UEN for precision
    snippets = _web_search(f'"{company_name}" Singapore UEN {uen}')

    # If sparse results, try a broader name search
    if len(snippets) < 2:
        snippets += _web_search(f'"{company_name}" Singapore company')

    try:
        return _extract_profile(company_name, uen, snippets)
    except Exception:
        # Fallback: minimal profile from user input alone
        return {
            "name": company_name,
            "uen": uen,
            "industry": "Unknown",
            "description": f"{company_name} is a Singapore-registered company.",
            "size_estimate": "SME",
            "key_activities": [],
        }
