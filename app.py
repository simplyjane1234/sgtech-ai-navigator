import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

load_dotenv()
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

from advisor import get_recommendation, get_topic_opener
from company_lookup import lookup_company

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SGTech AI Navigator",
    page_icon="🧭",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in {
    "onboarding_complete": False,
    "company": None,
    "topic_mode": None,       # "tools" | "grants" | "events" | "membership" | None
    "awaiting_topic": False,  # show topic selection buttons
    "messages": [],
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧭 SGTech AI Navigator")
    st.caption("Helping Singapore SMEs adopt AI with confidence.")
    st.divider()

    if st.session_state.company:
        co = st.session_state.company
        st.markdown("**Your company**")
        st.markdown(f"**{co['name']}**")
        st.caption(f"UEN: {co['uen']}  ·  {co['industry']}  ·  {co['size_estimate']}")
        st.divider()

    if st.session_state.onboarding_complete:
        if st.button("↩ Start over", use_container_width=True):
            for k in ["onboarding_complete", "company", "topic_mode",
                      "awaiting_topic", "messages", "pending_question"]:
                st.session_state[k] = {"onboarding_complete": False,
                                        "company": None,
                                        "topic_mode": None,
                                        "awaiting_topic": False,
                                        "messages": [],
                                        "pending_question": None}[k]
            st.rerun()

    st.divider()
    st.caption("Powered by SGTech · Built on OpenAI")


# ---------------------------------------------------------------------------
# Helper — add assistant message and rerun
# ---------------------------------------------------------------------------

def _assistant_reply(text: str):
    st.session_state.messages.append({"role": "assistant", "content": text})


def _clean_message(text: str) -> str:
    """Strip [OPTIONS: ...] tags from text before displaying or passing as history."""
    return re.sub(r'\n?\[OPTIONS:[^\]]+\]', '', text, flags=re.IGNORECASE).strip()


def _extract_quick_replies(text: str) -> list[str]:
    """
    Extract clickable quick-reply options from an assistant message.
    Detects two patterns:
      1. [OPTIONS: Choice A | Choice B | Choice C]  — explicit tag from LLM
      2. Numbered bold options: 1. **Title** ...     — choice lists
    """
    # Pattern 1: explicit [OPTIONS: A | B | C] tag
    m = re.search(r'\[OPTIONS:\s*([^\]]+)\]', text, re.IGNORECASE)
    if m:
        opts = [o.strip() for o in m.group(1).split('|') if o.strip()]
        if 2 <= len(opts) <= 4:
            return opts

    # Pattern 2: numbered bold options — 1. **Title**
    opts = re.findall(r'\d+\.\s+\*\*([^*\n:]+)', text)
    if not opts:
        opts = re.findall(r'^\d+\.\s+([^\n*:]{5,60})(?=\n|:|$)', text, re.MULTILINE)
    opts = [o.strip() for o in opts if o.strip()]
    return opts if 2 <= len(opts) <= 5 else []


def _handle_user_query(query: str):
    """Shared logic: append user message, call advisor, append and render reply."""
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history = [
        {"role": m["role"], "content": _clean_message(m["content"])}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = get_recommendation(
                    query=query,
                    history=history,
                    topic_mode=st.session_state.topic_mode,
                    company=st.session_state.company,
                )
            except Exception as e:
                response = f"Sorry, something went wrong: {e}"
        st.markdown(response)

    _assistant_reply(response)
    st.rerun()


def _handle_topic_switch(mode: str):
    """Switch to a new topic: generate opener and reset chat for that topic."""
    st.session_state.topic_mode = mode
    st.session_state.awaiting_topic = False
    with st.spinner("Loading..."):
        opener = get_topic_opener(mode, st.session_state.company)
    _assistant_reply(opener)
    st.rerun()


# ---------------------------------------------------------------------------
# Step 1 — Onboarding form
# ---------------------------------------------------------------------------

if not st.session_state.onboarding_complete:
    st.markdown("## Welcome to the SGTech AI Navigator 🧭")
    st.markdown(
        "I help Singapore SMEs find the right AI tools, grants, and starter kits. "
        "Let's start with a few details about your company."
    )
    st.divider()

    with st.form("onboarding_form"):
        company_name = st.text_input("Company name", placeholder="e.g. Bright Bakery Pte Ltd")
        uen = st.text_input(
            "UEN (Unique Entity Number)",
            placeholder="e.g. 202312345A",
            help="Your 9–10 character Singapore business registration number.",
        )
        submitted = st.form_submit_button("Look up my company →", use_container_width=True)

    if submitted:
        if not company_name.strip() or not uen.strip():
            st.warning("Please enter both your company name and UEN.")
        else:
            with st.spinner("Looking up your company online..."):
                profile = lookup_company(company_name.strip(), uen.strip().upper())

            st.session_state.company = profile
            st.session_state.onboarding_complete = True
            st.session_state.awaiting_topic = True

            activities = ", ".join(profile.get("key_activities") or []) or "not listed"
            greeting = (
                f"Great — here's what I found about **{profile['name']}**:\n\n"
                f"- **Industry:** {profile['industry']}\n"
                f"- **About:** {profile['description']}\n"
                f"- **Key activities:** {activities}\n\n"
                f"**What would you like to explore today?**"
            )
            st.session_state.messages = [{"role": "assistant", "content": greeting}]
            st.rerun()


# ---------------------------------------------------------------------------
# Step 2 — Topic selection + guided chat
# ---------------------------------------------------------------------------

else:
    st.markdown("## SGTech AI Navigator 🧭")

    # Render chat history (strip [OPTIONS:] tags from display)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(_clean_message(message["content"]))

    # ── Quick-reply buttons — appear when last assistant message has options ──
    if st.session_state.topic_mode and st.session_state.messages:
        last = st.session_state.messages[-1]
        if last["role"] == "assistant":
            quick_replies = _extract_quick_replies(last["content"])
            if quick_replies:
                # Always ensure a fallback escape option is present
                fallback = "I want to ask something else"
                if not any("else" in o.lower() or "different" in o.lower() for o in quick_replies):
                    quick_replies = quick_replies + [fallback]
                qcols = st.columns(len(quick_replies))
                for i, (col, option) in enumerate(zip(qcols, quick_replies), 1):
                    with col:
                        if st.button(option, key=f"qr_{i}", use_container_width=True):
                            _handle_user_query(option)
                st.caption("Or type your own answer below ↓")

    # ── Topic buttons ──
    # Full labels for the front page (vertical stacked)
    ALL_TOPICS = {
        "🛠 AI Tools":          "tools",
        "💰 Grants":            "grants",
        "🎯 AI Readiness":      "readiness",
        "📅 SGTech Events":     "events",
        "🤝 SGTech Membership": "membership",
    }
    # Short labels for the in-conversation nav row
    NAV_LABELS = {
        "🛠 Tools":      "tools",
        "💰 Grants":     "grants",
        "🎯 Readiness":  "readiness",
        "📅 Events":     "events",
        "🤝 Membership": "membership",
    }

    st.divider()

    if st.session_state.awaiting_topic:
        # Front page — large stacked full-width buttons
        for label, mode in ALL_TOPICS.items():
            if st.button(label, use_container_width=True, key=f"select_{mode}"):
                st.session_state.messages.append({"role": "user", "content": label})
                _handle_topic_switch(mode)
    else:
        # In conversation — compact horizontal row
        cols = st.columns(5)
        for col, (label, mode) in zip(cols, NAV_LABELS.items()):
            with col:
                is_active = st.session_state.topic_mode == mode
                if st.button(
                    label,
                    use_container_width=True,
                    key=f"nav_{mode}",
                    type="primary" if is_active else "secondary",
                    disabled=is_active,
                ):
                    st.session_state.messages.append({"role": "user", "content": label})
                    _handle_topic_switch(mode)

    # ── Chat input (active after a topic is chosen) ──
    if st.session_state.topic_mode:
        if query := st.chat_input("Type your answer or question..."):
            _handle_user_query(query)
    else:
        st.chat_input("Select a topic above to begin...", disabled=True)

    # ── Feedback button ──
    st.divider()
    st.markdown(
        "**Found this useful? Share your feedback** — it takes 2 minutes and helps us improve.  \n"
        "[📝 Take the survey →](https://forms.office.com/r/6NSM0xraFf)",
        unsafe_allow_html=False,
    )
