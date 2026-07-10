import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import get_session, Lead, QueryLog

db = get_session()

print("=== Leads ===")
for lead in db.query(Lead).all():
    print(f"  #{lead.id}  {lead.name}  {lead.phone}  {lead.email}")

print("\n=== Query Logs ===")
for q in db.query(QueryLog).all():
    print(f"  lead_id={q.lead_id}  Q: {q.question[:60]!r}")
    print(f"    sources: {q.sources}")

db.close()
