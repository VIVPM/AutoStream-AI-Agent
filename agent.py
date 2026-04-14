import os
from typing import Annotated, TypedDict
from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from rag import retrieve_context
from tools import mock_lead_capture


# ── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str  # "greeting", "inquiry", "high_intent"
    lead_info: dict  # collected lead fields: name, email, platform


# ── Gemini Client ────────────────────────────────────────────────────────────

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _client


def call_llm(system_instruction: str, messages: list, temperature: float = 0.3) -> str:
    """Call Gemini LLM and return the text response."""
    client = get_client()

    # Build conversation contents from LangChain messages
    contents = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            contents.append(types.Content(role="user", parts=[types.Part(text=msg.content)]))
        elif isinstance(msg, AIMessage):
            contents.append(types.Content(role="model", parts=[types.Part(text=msg.content)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )
    return response.text


# ── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI sales assistant for AutoStream, a SaaS platform that provides automated video editing tools for content creators.

Your responsibilities:
1. Greet users warmly and help them learn about AutoStream.
2. Answer product, pricing, and policy questions accurately using ONLY the provided context.
3. Detect when a user shows high intent to sign up or purchase.
4. When high intent is detected, collect the user's details (name, email, creator platform) ONE BY ONE before capturing the lead.

INTENT CLASSIFICATION RULES:
- "greeting": User sends a casual greeting like "hi", "hello", "hey", "good morning" without any product-related question.
- "inquiry": User asks about pricing, features, plans, policies, or general product questions.
- "high_intent": User expresses readiness to sign up, purchase, try, subscribe, or get started. Examples: "I want to sign up", "I'd like to try the Pro plan", "How do I get started?", "Sign me up", "I want to subscribe".

LEAD COLLECTION RULES:
- When you detect high intent, start collecting lead information.
- Ask for details ONE AT A TIME in this order: name, then email, then creator platform.
- Do NOT ask for multiple details in a single message.
- Only call the mock_lead_capture tool AFTER you have collected ALL THREE values (name, email, platform).
- NEVER call the tool with placeholder or assumed values.

RESPONSE GUIDELINES:
- Keep responses concise and friendly.
- When answering product questions, use the provided context from the knowledge base.
- Do not make up information not in the context.
- ONLY answer what the user asked. Do NOT volunteer extra information that was not requested. For example, if the user asks about pricing, only talk about pricing — do not mention refund policies or support unless they ask.
"""


# ── Node Functions ───────────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """Classify the user's intent from their latest message."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    lead_info = state.get("lead_info", {})

    # If we're in the middle of collecting lead info, stay in high_intent
    if lead_info and any(v is not None for v in lead_info.values()):
        return {**state, "intent": "high_intent"}

    classification_instruction = """Classify the user's intent into exactly one of these categories:
- "greeting": casual greeting with no product question
- "inquiry": asking about product, pricing, features, or policies
- "high_intent": expressing desire to sign up, purchase, try, subscribe, or get started

Respond with ONLY the category name, nothing else."""

    response = call_llm(
        system_instruction=classification_instruction,
        messages=[HumanMessage(content=f"User message: {last_message}")],
        temperature=0.0,
    )
    intent = response.strip().strip('"').lower()

    if intent not in ["greeting", "inquiry", "high_intent"]:
        intent = "inquiry"

    return {**state, "intent": intent}


def handle_greeting(state: AgentState) -> AgentState:
    """Handle casual greetings."""
    response = call_llm(
        system_instruction=SYSTEM_PROMPT,
        messages=state["messages"],
    )
    return {"messages": [AIMessage(content=response)]}


def handle_inquiry(state: AgentState) -> AgentState:
    """Handle product/pricing inquiries using RAG."""
    last_message = state["messages"][-1].content
    context = retrieve_context(last_message)

    response = call_llm(
        system_instruction=SYSTEM_PROMPT + f"\n\nKNOWLEDGE BASE CONTEXT:\n{context}",
        messages=state["messages"],
    )
    return {"messages": [AIMessage(content=response)]}


def handle_high_intent(state: AgentState) -> AgentState:
    """Handle high-intent users — collect lead info and capture lead."""
    lead_info = state.get("lead_info", {})
    last_message = state["messages"][-1].content

    # Determine what info we still need
    if "name" not in lead_info or lead_info["name"] is None:
        if lead_info.get("_asked_name"):
            lead_info["name"] = last_message.strip()
            lead_info["_asked_email"] = True
            response = f"Thanks, {lead_info['name']}! Could you please share your email address?"
            return {"messages": [AIMessage(content=response)], "lead_info": lead_info}
        else:
            lead_info["_asked_name"] = True
            context = retrieve_context(last_message)
            response = call_llm(
                system_instruction=SYSTEM_PROMPT + f"\n\nKNOWLEDGE BASE CONTEXT:\n{context}\n\nThe user has shown high intent to sign up. Acknowledge their interest enthusiastically and ask for their name to get started. Keep it brief.",
                messages=state["messages"],
            )
            return {"messages": [AIMessage(content=response)], "lead_info": lead_info}

    elif "email" not in lead_info or lead_info["email"] is None:
        if lead_info.get("_asked_email"):
            lead_info["email"] = last_message.strip()
            lead_info["_asked_platform"] = True
            response = "Great! And which platform do you primarily create content on? (e.g., YouTube, Instagram, TikTok)"
            return {"messages": [AIMessage(content=response)], "lead_info": lead_info}
        else:
            lead_info["_asked_email"] = True
            response = "Could you please share your email address?"
            return {"messages": [AIMessage(content=response)], "lead_info": lead_info}

    elif "platform" not in lead_info or lead_info["platform"] is None:
        if lead_info.get("_asked_platform"):
            lead_info["platform"] = last_message.strip()
            # All info collected — call the tool
            tool_result = mock_lead_capture.invoke({
                "name": lead_info["name"],
                "email": lead_info["email"],
                "platform": lead_info["platform"],
            })
            response = (
                f"Awesome! You're all set, {lead_info['name']}!\n\n"
                f"**Lead Captured:**\n"
                f"- **Name:** {lead_info['name']}\n"
                f"- **Email:** {lead_info['email']}\n"
                f"- **Platform:** {lead_info['platform']}\n\n"
                f"Our team will reach out to you shortly to help you get started with AutoStream. "
                f"Welcome aboard!"
            )
            return {"messages": [AIMessage(content=response)], "lead_info": {}}
        else:
            lead_info["_asked_platform"] = True
            response = "Great! And which platform do you primarily create content on? (e.g., YouTube, Instagram, TikTok)"
            return {"messages": [AIMessage(content=response)], "lead_info": lead_info}

    return {"messages": state["messages"], "lead_info": lead_info}


# ── Routing ──────────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate handler based on classified intent."""
    intent = state.get("intent", "greeting")
    if intent == "greeting":
        return "handle_greeting"
    elif intent == "high_intent":
        return "handle_high_intent"
    else:
        return "handle_inquiry"


# ── Build Graph ──────────────────────────────────────────────────────────────

def build_agent():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_inquiry", handle_inquiry)
    graph.add_node("handle_high_intent", handle_high_intent)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_by_intent)
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_inquiry", END)
    graph.add_edge("handle_high_intent", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ── Run Agent ────────────────────────────────────────────────────────────────

def run_agent(agent, user_input: str, config: dict, lead_info: dict = None):
    """Run the agent with a user message and return the response."""
    if lead_info is None:
        lead_info = {}

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)], "lead_info": lead_info},
        config=config,
    )

    ai_message = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            ai_message = msg.content
            break

    return ai_message, result.get("lead_info", {})
