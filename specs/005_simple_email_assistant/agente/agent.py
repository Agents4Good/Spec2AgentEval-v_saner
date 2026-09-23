import os
import re
from typing import Any, Dict, Tuple

try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception:  # pragma: no cover - optional dependency during import
    StateGraph = None
    END = None

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency during import
    ChatOpenAI = None


VALID_CLASSIFICATIONS = {"ignore", "notify", "respond", "unknown"}


def _normalize_email_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for key in ("email_text", "body", "text", "content", "message", "email", "input"):
            if key in raw and raw[key] is not None:
                return str(raw[key]).strip()
        return str(raw).strip()
    return str(raw).strip()


def _safe_llm_classify(email_text: str) -> str:
    if ChatOpenAI is None:
        return ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt = (
            "Classify the email into exactly one of: ignore, notify, respond. "
            "Use only the email content. Return only the label.\n\nEmail:\n"
            f"{email_text[:4000]}"
        )
        response = llm.invoke(prompt)
        label = str(response.content).strip().lower()
        if label in VALID_CLASSIFICATIONS:
            return label
        normalized = re.sub(r"[^a-z]", "", label)
        if normalized in {"ignore", "notify", "respond"}:
            return normalized
    except Exception:
        return ""
    return ""


def _classify_email(email_text: str) -> Tuple[str, str]:
    llm_label = _safe_llm_classify(email_text)
    if llm_label:
        if llm_label == "ignore":
            return "ignore", "The message appears to be promotional, spam, or irrelevant marketing content."
        if llm_label == "notify":
            return "notify", "The email contains urgent financial or legal information requiring attention."
        if llm_label == "respond":
            return "respond", "The email contains a direct request, question, or scheduling need that merits a reply."

    text = email_text.lower()
    if not text:
        return "unknown", "No email content was provided for classification."

    marketing_markers = [
        "unsubscribe", "limited time", "special offer", "buy now", "sale", "promo",
        "newsletter", "marketing", "click here", "free trial", "claim your reward",
        "exclusive deal", "new product", "incredible offer", "spam"
    ]
    financial_markers = [
        "invoice", "payment overdue", "late fee", "urgent action required", "cease and desist",
        "lawsuit", "compliance", "legal notice", "bank transfer", "wire transfer",
        "suspicious activity", "account locked", "regulatory", "settlement", "litigation"
    ]
    response_markers = [
        "meeting", "schedule", "appointment", "can we meet", "could we", "would you be available",
        "question", "customer support", "need help", "help me", "interested in", "when are you free",
        "please confirm", "reply by", "i would like to", "can you", "could you"
    ]

    if any(marker in text for marker in marketing_markers):
        return "ignore", "The message looks like promotional or irrelevant marketing content."

    if any(marker in text for marker in financial_markers):
        return "notify", "The email describes a time-sensitive financial or legal matter that should be escalated."

    if any(marker in text for marker in response_markers):
        return "respond", "The email asks a direct question or requests scheduling, which warrants a response."

    if re.search(r"\b(hello|hi|thanks|thank you|good morning|good afternoon)\b", text):
        return "respond", "The email is conversational and likely expects a direct reply."

    return "ignore", "The message does not match the criteria for urgency or direct response and is treated as non-actionable."


def _build_draft_response(email_text: str) -> str:
    normalized = email_text.strip()
    if not normalized:
        return "Thank you for your email. I will review your request and get back to you shortly."
    lower = normalized.lower()
    if any(token in lower for token in ["meeting", "schedule", "appointment", "call", "demo"]):
        return (
            "Hi,\n\nThank you for your message and for your interest in scheduling a conversation. "
            "I am happy to coordinate a time that works for both of us. Please share your preferred dates or times, "
            "and I will do my best to accommodate them.\n\nBest regards,\n[Your Name]"
        )
    if any(token in lower for token in ["question", "help", "support", "issue", "problem"]):
        return (
            "Hi,\n\nThank you for reaching out. I have reviewed your message and would be glad to help. "
            "Please share any additional details that would help me address your question accurately, and I will get back to you promptly.\n\nBest regards,\n[Your Name]"
        )
    return (
        "Hi,\n\nThank you for your email. I appreciate your time and will review the details you shared. "
        "I will follow up as soon as possible with the next steps.\n\nBest regards,\n[Your Name]"
    )


def agent(state: Dict[str, Any] | str) -> Dict[str, Any]:
    """Process an incoming email and return a structured classification result."""
    try:
        email_text = _normalize_email_text(state)
        if not email_text:
            raise ValueError("No email text was provided.")

        classification, message = _classify_email(email_text)
        draft_response = None
        if classification == "respond":
            draft_response = _build_draft_response(email_text)

        return {
            "action": "process_email",
            "classification": classification,
            "status": "success",
            "message": message,
            "draft_response": draft_response,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        return {
            "action": "process_email",
            "classification": "unknown",
            "status": "failure",
            "message": f"Failed to process email: {exc}",
            "draft_response": None,
        }


if StateGraph is not None:
    workflow = StateGraph(dict)
    workflow.add_node("classify_email", agent)
    workflow.set_entry_point("classify_email")
    workflow.add_edge("classify_email", END)
    app = workflow.compile()
else:  # pragma: no cover - fallback when langgraph is absent
    app = None


__all__ = ["agent", "app"]
