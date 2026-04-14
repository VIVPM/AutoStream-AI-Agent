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

## Architecture Explanation

This project uses **LangGraph** as the orchestration framework for building a stateful, multi-step conversational agent. LangGraph was chosen over AutoGen because it provides explicit control over state transitions through a directed graph model, which maps naturally to the intent-based conversation flow required (greeting → inquiry → high-intent → lead capture). Unlike AutoGen's multi-agent paradigm, LangGraph keeps the architecture simple with a single agent and clearly defined nodes for each stage of the workflow.

**State management** is handled through LangGraph's built-in `MemorySaver` checkpointer, which persists conversation state across multiple turns using a thread-based approach. The `AgentState` TypedDict tracks three key pieces of information: the full message history (for conversational context), the classified intent (greeting, inquiry, or high_intent), and the lead information dictionary (name, email, platform). This state flows through the graph — starting at an intent classification node, routing conditionally to the appropriate handler, and maintaining lead collection progress across turns. The checkpointer ensures memory persists across 5–6+ conversation turns without any external database.

**RAG** is implemented using ChromaDB as the vector store with Google's `gemini-embedding-001` model (384 dimensions). The knowledge base is stored in a local JSON file and embedded into ChromaDB at startup, enabling semantic search for accurate product and pricing responses.

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

| Criteria | Implementation |
|----------|---------------|
| Intent Detection | LLM-based classification into greeting/inquiry/high_intent |
| RAG | ChromaDB + gemini-embedding-001 (384d) over local JSON knowledge base |
| State Management | LangGraph MemorySaver with thread-based checkpointing |
| Tool Calling | mock_lead_capture called only after collecting all 3 fields |
| Code Clarity | Modular design — separate files for agent, RAG, tools, and UI |
| Deployability | Streamlit UI + WhatsApp webhook architecture documented |
