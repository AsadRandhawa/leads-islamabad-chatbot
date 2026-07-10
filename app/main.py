from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.exc import IntegrityError

from app.db import Lead, QueryLog, get_session, init_db
from app.rag import RagEngine

load_dotenv()

app = FastAPI(title="Leads University Islamabad Campus Chatbot")

# TODO before going live: replace "*" with your real WordPress domain,
# e.g. ["https://leads.edu.pk"] — wide-open CORS is fine for local testing
# only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[RagEngine] = None


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
    """Captures name/phone/email before someone starts chatting. No login —
    just a contact record so the admissions team can follow up later.
    If someone already exists with the same email or phone (e.g. they
    reopened the form in a new browser/incognito window), reuse that
    record instead of creating a duplicate."""
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


class ChatRequest(BaseModel):
    message: str
    lead_id: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = engine.answer(req.message)

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


@app.get("/")
def widget():
    return FileResponse(Path(__file__).parent.parent / "frontend" / "widget.html")
