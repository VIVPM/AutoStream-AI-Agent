import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from agent import build_agent, run_agent


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    layout="centered"
)

# ── API Key Validation ──────────────────────────────────────────────────────

if not os.getenv("GEMINI_API_KEY"):
    st.error(
        "GEMINI_API_KEY is not set. Create a `.env` file in the project root "
        "with `GEMINI_API_KEY=<your_key>` and restart the app."
    )
    st.stop()

st.title("🎬 AutoStream AI Assistant")
st.caption("Your AI-powered guide to automated video editing tools for content creators.")

# ── Session State Init ───────────────────────────────────────────────────────

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lead_info" not in st.session_state:
    st.session_state.lead_info = {}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "autostream-session-1"

# Agent config with thread_id for memory persistence
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Display Chat History ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Welcome Message ──────────────────────────────────────────────────────────

if not st.session_state.messages:
    welcome = (
        "Hi there! Welcome to **AutoStream** — the automated video editing platform "
        "built for content creators. 🎬\n\n"
        "I can help you with:\n"
        "- **Pricing & Plans** — Learn about our Basic and Pro plans\n"
        "- **Features** — See what AutoStream can do for you\n"
        "- **Getting Started** — Sign up and start editing!\n\n"
        "How can I help you today?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    with st.chat_message("assistant"):
        st.markdown(welcome)

# ── Chat Input ───────────────────────────────────────────────────────────────

if user_input := st.chat_input("Type your message..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, lead_info = run_agent(
                st.session_state.agent,
                user_input,
                config,
                st.session_state.lead_info
            )
            st.session_state.lead_info = lead_info
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About AutoStream")
    st.markdown("""
    **AutoStream** is a SaaS platform providing automated video editing tools for content creators.

    **Plans:**
    | Plan | Price |
    |------|-------|
    | Basic | $29/month |
    | Pro | $79/month |

    **Key Features:**
    - AI-powered video editing
    - Auto-captions (Pro)
    - 4K export (Pro)
    - Multi-platform support
    """)

    st.divider()

    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.lead_info = {}
        st.session_state.thread_id = f"autostream-session-{hash(str(st.session_state))}"
        st.session_state.agent = build_agent()
        st.rerun()
