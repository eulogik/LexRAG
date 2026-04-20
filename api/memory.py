import sqlite3
import json
import os
from datetime import datetime
from sqlite_utils import Database

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lexrag.db")

def setup_db():
    db = Database(DB_PATH)
    if "conversations" not in db.table_names():
        db["conversations"].create({
            "id": int,
            "session_id": str,
            "role": str,
            "content": str,
            "sources": str,
            "timestamp": str,
            "provider": str
        }, pk="id")
    if "sessions" not in db.table_names():
        db["sessions"].create({
            "session_id": str,
            "name": str,
            "created_at": str,
            "updated_at": str
        }, pk="session_id")
    return db

def update_session_name(session_id: str, name: str):
    db = setup_db()
    now = datetime.now().isoformat()
    try:
        db["sessions"].upsert({
            "session_id": session_id,
            "name": name,
            "created_at": now,
            "updated_at": now
        }, pk="session_id")
    except Exception:
        db["sessions"].update(session_id, {"name": name, "updated_at": now})

def get_session_name(session_id: str) -> str:
    db = setup_db()
    try:
        row = db["sessions"].get(session_id)
        return row["name"] if row else session_id[:8]
    except Exception:
        return session_id[:8]

def save_message(session_id: str, role: str, content: str, sources: list = None, provider: str = None):
    db = setup_db()
    db["conversations"].insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": json.dumps(sources) if sources else "[]",
        "timestamp": datetime.now().isoformat(),
        "provider": provider
    })

def get_history(session_id: str, limit: int = 10):
    db = setup_db()
    rows = db["conversations"].rows_where(
        "session_id = ?", [session_id], order_by="timestamp DESC", limit=limit
    )
    history = list(rows)
    history.reverse()
    return [{"role": r["role"], "content": r["content"]} for r in history]

def get_history_full(session_id: str, limit: int = 100):
    """Returns full message objects including sources for UI rendering."""
    db = setup_db()
    rows = db["conversations"].rows_where(
        "session_id = ?", [session_id], order_by="timestamp ASC", limit=limit
    )
    result = []
    for r in rows:
        sources = []
        try:
            sources = json.loads(r.get("sources", "[]"))
        except Exception:
            pass
        result.append({
            "role": r["role"],
            "content": r["content"],
            "sources": sources,
            "provider": r.get("provider"),
            "timestamp": r.get("timestamp")
        })
    return result

def list_sessions():
    db = setup_db()
    if "conversations" not in db.table_names():
        return []
    rows = list(db.query("""
        SELECT c.session_id,
               COALESCE(s.name, c.session_id) as name,
               MAX(c.timestamp) as last_active,
               MIN(c.timestamp) as created_at,
               COUNT(*) as message_count,
               (SELECT content FROM conversations c2 WHERE c2.session_id = c.session_id AND c2.role = 'user' ORDER BY c2.timestamp ASC LIMIT 1) as preview
        FROM conversations c
        LEFT JOIN sessions s ON s.session_id = c.session_id
        GROUP BY c.session_id
        ORDER BY last_active DESC
    """))
    return rows

def delete_session(session_id: str):
    db = setup_db()
    db["conversations"].delete_where("session_id = ?", [session_id])
    try:
        db["sessions"].delete(session_id)
    except Exception:
        pass
