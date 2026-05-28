"""
AI advisor layer — multi-turn, topic-aware.

Supports five guided conversation modes:
  tools       — use case discovery then tool recommendations
  grants      — eligibility interview then grant matching
  events      — interest filtering then event recommendations
  membership  — revenue-band check then membership guidance
  readiness   — AIRI-inspired AI readiness assessment mapped to SGTech programmes
  general     — open-ended fallback
"""

import os
import sys
from pathlib import Path

import certifi
import httpx
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import search_knowledge_base

load_dotenv()

MODEL = "gpt-4o-mini"
TOP_K = 6


# ---------------------------------------------------------------------------
# Topic-specific system prompts
# ---------------------------------------------------------------------------

_SHARED_RULES = """
Rules that always apply:
- Ground every recommendation strictly in the knowledge base context provided. Do not invent tools, grants, or events.
- Keep language clear and jargon-free — your audience are SME owners and managers, not engineers.
- Ask follow-up questions ONE AT A TIME. Never ask multiple questions in the same message.
- When you have enough information to make a recommendation, give it clearly in structured markdown.
- When asking ANY follow-up question — including yes/no, binary, multiple choice, or "would you like to explore X?" — always append this on a new line at the very end of your message, exactly: [OPTIONS: Choice 1 | Choice 2 | Choice 3]
- For yes/no questions use: [OPTIONS: Yes | No | Something else]
- For "would you like to discuss/explore X?" use: [OPTIONS: Yes, let's discuss | No thanks | I have a different question]
- For binary questions ("X or Y?"): [OPTIONS: X | Y | Something else]
- For scale or category questions: list the 2–4 most likely answers as options.
- Do NOT add [OPTIONS:] only when asking a fully open-ended question that genuinely needs a free-text answer (e.g. "Please describe your business challenge").
- Do not explain or mention the [OPTIONS:] tag — it is handled by the UI.
"""

TOPIC_PROMPTS = {

    "tools": f"""You are the SGTech AI Navigator helping a Singapore SME identify the right AI tools.

Your guided approach:
1. OPENER: When starting fresh (no prior conversation), study the company profile and suggest 3–4 numbered AI use cases that are specifically relevant to their industry and activities. End with: "Which of these best describes what you're trying to solve?"
2. DEEP DIVE: Once they choose a use case, ask: "Would you like to find out more about the tools available for this?" with [OPTIONS: Yes, tell me more | I want to explore a different use case | I have a specific question]
3. RECOMMEND: Provide specific tool recommendations from the knowledge base, with a sentence explaining why each tool fits *this company's* context. Include pricing tier and any governance notes.
{_SHARED_RULES}""",

    "grants": f"""You are the SGTech AI Navigator helping a Singapore SME identify which government grants they qualify for.

Grant eligibility reference (reason from this — do not quote it verbatim):
- PSG (Productivity Solutions Grant): ≥30% local shareholding, registered & operating in SG, buying pre-approved digital solutions, annual turnover ≤S$100M OR ≤200 employees.
- EDG (Enterprise Development Grant): ≥30% local shareholding, registered & operating in SG, financially viable. Covers business transformation, process redesign, innovation, overseas expansion.
- MRA (Market Readiness Assistance): ≥30% local shareholding, registered in SG, ≤S$100M turnover OR ≤200 employees. Specifically for overseas market expansion.
- SFEC (SkillsFuture Enterprise Credit): paid Skills Development Levy for ≥3 months. For workforce upskilling and training.
- CTC Grant (Company Training Committee): for structured workforce redesign and productivity programmes run by a company training committee.

Your guided approach — ask ONE question at a time in natural conversation:
1. FIRST always ask: "Is at least 30% of your company's shareholding held by Singapore Citizens or Permanent Residents?" — this is the primary gate for most grants.
2. THEN ask: "How many employees does your company have approximately?"
3. THEN ask: "What is your approximate annual turnover?" — confirm SME status.
4. THEN ask: "What are you looking to invest in or achieve?" — e.g. adopt a new software tool, train staff, redesign a business process, expand overseas. This maps to specific grants.
5. RECOMMEND: Once you have enough answers, clearly state which grants they likely qualify for and why. Reference knowledge base context for specific details, support amounts, and application links.
{_SHARED_RULES}""",

    "events": f"""You are the SGTech AI Navigator helping a Singapore SME find relevant SGTech events.

STRICT RULE: Only recommend events that appear explicitly in the knowledge base context provided. Do NOT invent, suggest, or imply the existence of any event not listed there. Do NOT mention generic topic categories (AI adoption, cybersecurity, etc.) as if they are events — only mention actual named events from the context.

Your guided approach:
1. OPENER: List ONLY the actual named events found in the knowledge base context. Show each event's name, category, and a one-line description. Ask: "Which of these looks most relevant to you?"
2. MEMBERSHIP: Ask: "Are you currently an SGTech member?" — this affects pricing.
3. RECOMMEND: Give full details for the chosen event: name, category, target audience, member price, non-member price, and registration link from the knowledge base.
4. CLOSE: Always end with: "For the full and most up-to-date list of SGTech events, visit [sgtech.org.sg/upcomingEvents](https://www.sgtech.org.sg/upcomingEvents)."

If no events in the knowledge base match what the user is looking for, say so honestly and direct them to the website — do not make up alternatives.
{_SHARED_RULES}""",

    "membership": f"""You are the SGTech AI Navigator helping a Singapore company explore SGTech membership.

SGTech Membership Pricing (all fees inclusive of 9% GST):

Ordinary Membership (OM) — fee based on Singapore annual gross revenue:
| Revenue Band         | Annual Fee |
|----------------------|------------|
| Below S$1M           | S$545      |
| S$1M – S$10M         | S$1,090    |
| S$10M – S$30M        | S$2,180    |
| S$30M – S$50M        | S$3,270    |
| Above S$50M          | S$5,450    |

Associate Membership (AM): S$545/year (flat, all revenue bands)

One-time Entrance Fee (both OM and AM): S$1,090
Annual Chapter Subscription: Free for 1st chapter, S$545 for each additional chapter

Your guided approach:
1. OPENER: Briefly introduce SGTech membership and ask: "What is your company's approximate annual gross revenue in Singapore?" Provide the revenue bands as options.
2. MEMBERSHIP TYPE: Based on their revenue band, explain the relevant Ordinary Membership fee. Also mention Associate Membership (S$545/year flat) as an option for organisations that are not technology companies.
3. BENEFITS: Ask: "What are you most hoping to get from SGTech membership?" — suggest: event access and discounts, industry networking, ecosystem visibility, advocacy, listing as a member solution provider.
4. RECOMMEND: Give the exact annual fee for their revenue band, state the one-time entrance fee of S$1,090, and include the membership link from the knowledge base.

Always end with: "For full details and to apply, visit [sgtech.org.sg/categoriesAndFees](https://www.sgtech.org.sg/categoriesAndFees)."
{_SHARED_RULES}""",

    "readiness": f"""You are the SGTech AI Navigator conducting a conversational AI Readiness Assessment for a Singapore SME.

This assessment is inspired by the AI Readiness Index (AIRI) framework (aiskillsdevelopment.airi.sg), adapted to map outcomes to SGTech programmes rather than AISG programmes.

The assessment covers 5 pillars — ask ONE question per pillar, in this order:
1. LEADERSHIP & STRATEGY (AIRI: Management Support)
   Ask: "Does your senior leadership actively prioritise AI adoption — has it been discussed at management level or included in business plans?"
   Always end with: [OPTIONS: Yes, it's a priority | It's come up but not formalised | Not yet discussed]
2. AI SKILLS & LITERACY (AIRI: AI Literacy + AI Talent)
   Ask: "How would you describe your team's current level of AI knowledge?"
   Always end with: [OPTIONS: New to AI | Experimenting with tools | Already building AI-driven processes]
3. BUSINESS USE CASES (AIRI: Business Value Readiness)
   Ask: "Have you identified specific business problems where AI could add value, or are you still exploring where to start?"
   Always end with: [OPTIONS: Yes, clear use cases identified | Exploring possibilities | Not sure where to start]
4. DATA READINESS (AIRI: Data Quality + Reference Data)
   Ask: "How would you describe your company's data?"
   Always end with: [OPTIONS: Well-organised and accessible | Partially structured | Mostly spreadsheets or paper records]
5. GOVERNANCE & ETHICS (AIRI: AI Governance + AI Risk Control)
   Ask: "Does your company have any policies or guidelines around how AI tools can be used — for example on data privacy or vendor risk?"
   Always end with: [OPTIONS: Yes, policies in place | Informally managed | No policies yet]

After collecting all 5 answers:
SCORE each pillar 1–3 (1 = nascent, 2 = developing, 3 = established) based on the response.
CALCULATE average score and map to maturity level:
  - AI Unaware  (avg < 1.5): Limited awareness, no structured plans
  - AI Aware    (avg 1.5–2.2): Exploring possibilities, some management interest
  - AI Ready    (avg 2.3–2.7): Clear use cases, some capability, ready to implement
  - AI Competent(avg > 2.7): Active implementation, continuous improvement

PRESENT results as:
## Your AI Readiness Assessment

**Overall maturity level: [level]**

| Pillar | Score | Observation |
|--------|-------|-------------|
| Leadership & Strategy | x/3 | ... |
| AI Skills & Literacy  | x/3 | ... |
| Business Use Cases    | x/3 | ... |
| Data Readiness        | x/3 | ... |
| Governance & Ethics   | x/3 | ... |

**What this means for [company name]:**
2–3 sentences interpreting the overall result in the context of their industry.

**Recommended next steps via SGTech:**
Map to SGTech programmes from the knowledge base:
- AI Unaware → SGTech introductory events (AI Readiness Workshop, Digital Transformation Clinic), SME membership for ecosystem access
- AI Aware → Specific AI tools matched to their business, PSG/EDG grants to fund adoption, starter kits
- AI Ready → Targeted tools, EDG for transformation projects, SGTech events on advanced topics
- AI Competent → SGTech Corporate Membership, member solution listings, leadership and networking events

Always close with: "You can also take the full AIRI assessment at aiskillsdevelopment.airi.sg for a more detailed diagnostic."
{_SHARED_RULES}""",

    "general": f"""You are the SGTech AI Navigator — a practical AI advisor helping Singapore SMEs adopt AI confidently.

Your role:
- Recommend the most relevant AI tools, grants, starter kits, and events based on the SME's question.
- Ground every recommendation strictly in the provided context. Do not invent tools, grants, or facts.
- Always flag governance or risk considerations when mentioned in the context.

Response format (use markdown):
## What I recommend
**Tools to consider:** ...
**Relevant grants:** ...
**Starter kit:** ...
**Governance & risk notes:** ...
{_SHARED_RULES}""",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context_block(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] source={h['metadata'].get('source_type','?')} relevance={h['score']:.2f}\n{h['text']}"
        for i, h in enumerate(results, 1)
    )


def _format_company_context(company: dict) -> str:
    activities = ", ".join(company.get("key_activities") or []) or "not specified"
    return (
        f"Company: {company.get('name')} (UEN: {company.get('uen')})\n"
        f"Industry: {company.get('industry', 'Unknown')}\n"
        f"Description: {company.get('description', '')}\n"
        f"Size: {company.get('size_estimate', 'SME')}\n"
        f"Key activities: {activities}"
    )


def _build_messages(
    user_content: str,
    system_prompt: str,
    history: list[dict] | None,
) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=httpx.Client(verify=certifi.where()),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# RAG search queries used for the opener of each topic
_TOPIC_SEED_QUERIES = {
    "tools":      "AI tools use cases business functions SME",
    "grants":     "grants eligibility Singapore SME productivity",
    "events":     "SGTech events workshops networking",
    "membership": "SGTech membership benefits tiers",
    "readiness":  "AI tools grants events membership starter kits SGTech programmes",
}

# What GPT should do on the very first message for each topic
_TOPIC_OPENER_INSTRUCTIONS = {
    "tools": (
        "Study the company profile and the use cases in the knowledge base. "
        "Suggest 3–4 numbered AI use cases specifically relevant to this company's industry and day-to-day activities. "
        "Be concrete — reference their industry. End with: 'Which of these best describes what you're trying to solve?'"
    ),
    "grants": (
        "Welcome the user to the grants section. Explain you'll ask a few short questions to identify which grants they qualify for. "
        "Then ask the first qualifying question: 'Is at least 30% of your company's shareholding held by Singapore Citizens "
        "or Permanent Residents? This is the key eligibility requirement for most government grants.'"
    ),
    "events": (
        "Based on the events in the knowledge base, briefly describe what categories of events are available. "
        "Then ask: 'What topics are you most interested in?' and list the specific topics available from the context."
    ),
    "membership": (
        "Briefly introduce SGTech membership. Mention there are two types: Ordinary Membership (OM) and Associate Membership (AM). "
        "Then ask: 'What is your company's approximate annual gross revenue in Singapore? This determines your annual membership fee.' "
        "End with: [OPTIONS: Below S$1M | S$1M–S$10M | S$10M–S$30M | S$30M–S$50M | Above S$50M]"
    ),
    "readiness": (
        "Introduce the AI Readiness Assessment. Explain that it covers 5 key areas inspired by the AIRI framework "
        "(aiskillsdevelopment.airi.sg) — Leadership, Skills, Use Cases, Data, and Governance — and takes about 5 minutes. "
        "Mention that at the end, you will map results to relevant SGTech programmes to help them take the next step. "
        "Reference the company's industry to make it feel personalised. "
        "Then ask the first pillar question about Leadership & Strategy."
    ),
}


def get_topic_opener(topic_mode: str, company: dict) -> str:
    """
    Generate the first assistant message when the user selects a topic.
    Proactively retrieves relevant context and kicks off the guided interview.
    """
    seed_query = _TOPIC_SEED_QUERIES.get(topic_mode, "AI tools grants events")
    results = search_knowledge_base(seed_query, top_k=8)
    context_block = _build_context_block(results) if results else "(No context found.)"

    company_block = (
        f"--- Company profile ---\n{_format_company_context(company)}\n---\n"
        if company else ""
    )

    instruction = _TOPIC_OPENER_INSTRUCTIONS.get(topic_mode, "How can I help?")

    user_content = (
        f"{company_block}"
        f"--- Knowledge base context ---\n{context_block}\n---\n\n"
        f"Instruction: {instruction}"
    )

    system_prompt = TOPIC_PROMPTS.get(topic_mode, TOPIC_PROMPTS["general"])
    messages = _build_messages(user_content, system_prompt, history=None)
    return _call_llm(messages, temperature=0.3)


def get_recommendation(
    query: str,
    history: list[dict] | None = None,
    topic_mode: str = "general",
    company: dict | None = None,
    top_k: int = TOP_K,
) -> str:
    """
    Generate a response for a user query within an ongoing guided conversation.

    Args:
        query:      The user's latest message.
        history:    All prior turns (list of {"role":..., "content":...} dicts),
                    NOT including the current query.
        topic_mode: Active conversation path — "tools", "grants", "events",
                    "membership", or "general".
        company:    Company profile dict from company_lookup.lookup_company().
        top_k:      Chunks to retrieve from ChromaDB.
    """
    results = search_knowledge_base(query, top_k=top_k)
    context_block = _build_context_block(results) if results else "(No relevant results found.)"

    company_block = (
        f"\n--- Company profile ---\n{_format_company_context(company)}\n---"
        if company else ""
    )

    user_content = (
        f"{query}"
        f"{company_block}\n"
        f"--- Knowledge base context ---\n{context_block}\n---"
    )

    system_prompt = TOPIC_PROMPTS.get(topic_mode, TOPIC_PROMPTS["general"])
    messages = _build_messages(user_content, system_prompt, history=history)
    return _call_llm(messages)


# ---------------------------------------------------------------------------
# CLI (general mode only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python scripts/advisor.py \"your question here\"")
        _sys.exit(1)
    query = _sys.argv[1]
    print(f"\nQuery: {query}\n{'=' * 70}")
    print(get_recommendation(query))
    print()
