# AutoStream AI Agent — Social-to-Lead Agentic Workflow

A conversational AI agent for **AutoStream**, a SaaS platform that provides automated video editing tools for content creators. The agent identifies user intent, answers product questions using RAG, and captures leads through a structured workflow.

## How to Run Locally

### Prerequisites

- Python 3.9+
- Google API Key (for Gemini LLM and embeddings)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/assignment-servicehive.git
   cd assignment-servicehive
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variable**

   ```bash
   export GOOGLE_API_KEY="your_google_api_key_here"
   ```

   On Windows:

   ```bash
   set GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Run the application**

   ```bash
   streamlit run app.py
   ```

5. Open `http://localhost:8501` in your browser.

## Architecture

```
User Input (message)
        |
        v
  Streamlit UI (app.py)
        |
        v
  classify_intent
  (Gemini LLM)
        |
   intent?
   /    |    \
greet  inq  high_intent
  |     |        |
  |     |        v
  |     |   Collect lead info
  |     |   (name → email → platform)
  |     |        |
  |   RAG        v
  |  lookup  mock_lead_capture
  |   (ChromaDB + Gemini embeddings)
  |     |        |
   \    |       /
    v   v      v
   LangGraph END
   (MemorySaver checkpoint)
        |
        v
  Streamlit UI (reply)
        |
        v
      User
```

**classify_intent** — Classifies every message into `greeting`, `inquiry`, or `high_intent` using Gemini 2.5 Flash at `temperature=0.0`.

**handle_greeting** — Generates a warm welcome response. No RAG needed.

**handle_inquiry** — Retrieves the top-k relevant chunks from ChromaDB (via `retrieve_context`) and passes them as context to Gemini before responding.

**handle_high_intent** — Runs a stateful one-field-at-a-time collection loop (name → email → platform). Calls `mock_lead_capture` only after all three fields are collected.

**RAG Pipeline** — `knowledge_base.json` is embedded at startup using `gemini-embedding-001` (384d) into a local ChromaDB store. Semantic search retrieves the most relevant product/pricing context on demand.

**MemorySaver** — LangGraph's built-in checkpointer persists `AgentState` (messages + intent + lead_info) across turns using a `thread_id`, so no external database is needed.

## WhatsApp Deployment via Webhooks

To integrate this agent with WhatsApp, the following webhook-based architecture can be used:

1. **WhatsApp Business API**: Register with the WhatsApp Business Platform (via Meta) to get API access and a phone number.

2. **Webhook Server**: Deploy a FastAPI/Flask server that exposes a webhook endpoint (e.g., `POST /webhook`). Configure this URL in the Meta Developer Dashboard as the callback URL.

3. **Incoming Messages**: When a user sends a WhatsApp message, Meta sends a POST request to the webhook with the message payload (sender number, message text, timestamp).

4. **Agent Processing**: The webhook handler extracts the message text, passes it to the LangGraph agent (using the sender's phone number as the `thread_id` for session persistence), and receives the agent's response.

5. **Outgoing Messages**: The server sends the agent's response back via the WhatsApp Business API's `POST /messages` endpoint with the sender's number and response text.

6. **Session Management**: Each user's phone number serves as a unique thread ID, allowing the LangGraph checkpointer to maintain conversation state across multiple messages — just as it does in the Streamlit UI.

```
User (WhatsApp) → Meta Cloud API → Webhook (POST /webhook) → LangGraph Agent → WhatsApp API (reply) → User
```

This approach keeps the core agent logic unchanged — only the I/O layer (Streamlit vs. webhook) differs.

## Project Structure

```
├── app.py                  # Streamlit UI
├── agent.py                # LangGraph agent (intent detection, routing, lead capture)
├── rag.py                  # RAG pipeline (ChromaDB + Gemini embeddings)
├── tools.py                # Mock lead capture tool
├── knowledge_base.json     # AutoStream product knowledge base
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── README.md               # This file
```

## Evaluation Criteria Coverage

| Criteria         | Implementation                                                        |
| ---------------- | --------------------------------------------------------------------- |
| Intent Detection | LLM-based classification into greeting/inquiry/high_intent            |
| RAG              | ChromaDB + gemini-embedding-001 (384d) over local JSON knowledge base |
| State Management | LangGraph MemorySaver with thread-based checkpointing                 |
| Tool Calling     | mock_lead_capture called only after collecting all 3 fields           |
| Code Clarity     | Modular design — separate files for agent, RAG, tools, and UI         |
| Deployability    | Streamlit UI + WhatsApp webhook architecture documented               |
