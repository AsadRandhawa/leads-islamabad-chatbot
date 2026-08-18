import os
import secrets
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.exc import IntegrityError

from app.db import Lead, QueryLog, get_session, init_db
from app.rag import RagEngine

load_dotenv()

app = FastAPI(title="Leads University Islamabad Campus Chatbot")

# Restricted to the real domain(s) the widget is actually embedded on.
# CONFIRM THIS against the live site before deploying — view-source on
# isb.leads.edu.pk and check where the widget script/iframe actually
# points, since this was not independently verified.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://leads.edu.pk", "https://isb.leads.edu.pk"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[RagEngine] = None

# Shared secret for CareAgent's server-to-server calls (the /partner/chat
# route below). This is NOT the same protection as CORS — CORS only
# restricts browser-based calls, and server-to-server calls (like
# CareAgent's backend) never go through a browser, so they'd bypass CORS
# entirely. This header check is what actually gates /partner/chat.
#
# Read with .get(), not bracket access — if this isn't set in Railway,
# only /partner/chat should fail (with a clear 503), not the entire app on
# startup. A missing WhatsApp-integration secret should never take down
# the main website chatbot.
CAREAGENT_SHARED_SECRET = os.environ.get("CAREAGENT_SHARED_SECRET")


def verify_careagent_secret(x_careagent_secret: str = Header(None)):
    if not CAREAGENT_SHARED_SECRET:
        raise HTTPException(
            status_code=503,
            detail="CareAgent integration is not configured on this server.",
        )
    if not x_careagent_secret or not secrets.compare_digest(x_careagent_secret, CAREAGENT_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def startup():
    global engine
    engine = RagEngine()
    init_db()


class LeadCreate(BaseModel):
    name: str
    phone: str
    email: EmailStr


class LeadResponse(BaseModel):
    lead_id: int


@app.post("/leads", response_model=LeadResponse)
def create_lead(req: LeadCreate):
    """Called by the host page (via window.LEADS_CHAT_USER) to link a chat
    session to a known, logged-in visitor. Not a mandatory step — most
    visitors are anonymous prospects with no account yet, and the widget
    has no in-widget lead-capture form; chat works fine without this ever
    being called. If someone already exists with the same email or phone
    (e.g. they reopened the form in a new browser/incognito window), reuse
    that record instead of creating a duplicate."""
    db = get_session()
    try:
        name = req.name.strip()
        phone = req.phone.strip()
        email = str(req.email).strip()

        existing = (
            db.query(Lead)
            .filter((Lead.email.ilike(email)) | (Lead.phone == phone))
            .order_by(Lead.id.asc())
            .first()
        )
        if existing:
            return LeadResponse(lead_id=existing.id)

        lead = Lead(name=name, phone=phone, email=email)
        db.add(lead)
        try:
            db.commit()
        except IntegrityError:
            # Two requests for the same person landed at the same instant
            # and both passed the "does this exist" check above before
            # either committed. The unique constraint on phone/email caught
            # it — roll back and just return the row that won the race,
            # instead of surfacing a raw database error to the client.
            db.rollback()
            existing = (
                db.query(Lead)
                .filter((Lead.email.ilike(email)) | (Lead.phone == phone))
                .order_by(Lead.id.asc())
                .first()
            )
            if existing:
                return LeadResponse(lead_id=existing.id)
            raise
        db.refresh(lead)
        return LeadResponse(lead_id=lead.id)
    finally:
        db.close()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    lead_id: Optional[int] = None
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [turn.model_dump() for turn in req.history]
    result = engine.answer(req.message, history=history)

    # Log every question + answer for follow-up/engagement, tied to the
    # lead if we have one. This must NEVER break the actual chat response —
    # if a client sent a stale/invalid lead_id (e.g. from localStorage
    # pointing at a database that got reset), log it as anonymous instead
    # of failing the whole request.
    db = get_session()
    try:
        lead_id = req.lead_id
        if lead_id is not None and not db.query(Lead.id).filter(Lead.id == lead_id).first():
            lead_id = None

        db.add(QueryLog(
            lead_id=lead_id,
            question=req.message,
            answer=result["answer"],
            sources=", ".join(result["sources"]),
        ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

    return ChatResponse(**result)


@app.post("/partner/chat", response_model=ChatResponse, dependencies=[Depends(verify_careagent_secret)])
def partner_chat(req: ChatRequest):
    """Server-to-server endpoint for CareAgent's WhatsApp integration.
    Deliberately does NOT touch Lead/QueryLog — most website chats are
    already anonymous too (see create_lead's docstring above), so WhatsApp
    follows the same pattern rather than being a special case. Revisit
    this if/when lead-scoring behavior gets built for WhatsApp."""
    history = [turn.model_dump() for turn in req.history]
    result = engine.answer(req.message, history=history)
    return ChatResponse(**result)


@app.get("/")
def widget():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "widget.html")
