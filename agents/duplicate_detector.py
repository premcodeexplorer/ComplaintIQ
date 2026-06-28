"""Agent 3 -- Duplicate Detector.

Embeds complaint text with sentence-transformers (all-MiniLM-L6-v2) and stores
embeddings in a Supabase pgvector column. For each new complaint, queries
the top-k nearest neighbours from the same customer and returns the best match
above a similarity threshold.
"""
from __future__ import annotations

from typing import Any

from database import db

MODEL_NAME = "all-MiniLM-L6-v2"
DUP_THRESHOLD = 0.78  # cosine similarity

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    return _get_model().encode([text], normalize_embeddings=True)[0].tolist()


def index_complaint(complaint: dict[str, Any]) -> None:
    """Add a complaint to the vector store. Idempotent on `id`."""
    cid = complaint["id"]
    text = complaint.get("complaint_text", "") or ""
    vec = embed(text)
    
    if db.IS_POSTGRES:
        with db.connect() as c:
            db._exec(c, "UPDATE complaints SET embedding = %s WHERE id = %s", (vec, cid))


def find_duplicate(complaint: dict[str, Any], *, threshold: float = DUP_THRESHOLD,
                   top_k: int = 5) -> dict[str, Any]:
    """Return {is_duplicate, duplicate_of, similarity, neighbours[]}.

    Restricts the search to the same customer.
    """
    text = complaint.get("complaint_text", "") or ""
    qvec = embed(text)

    customer = complaint.get("customer_name") or ""
    cid = complaint.get("id") or ""
    
    if db.IS_POSTGRES:
        query = """
            SELECT id, customer_name, channel, date, category, location,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM complaints
            WHERE customer_name = %s AND id != %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (qvec, customer, cid, qvec, top_k)
        
        with db.connect() as c:
            rows = db._exec(c, query, params).fetchall()
        
        neighbours = []
        best = (None, 0.0)
        
        for row in rows:
            sim = float(row["similarity"])
            neighbours.append({
                "id": row["id"],
                "similarity": round(sim, 4),
                "meta": {
                    "customer_name": row["customer_name"],
                    "channel": row["channel"],
                    "date": row["date"],
                    "category": row["category"],
                    "location": row["location"],
                }
            })
            if sim > best[1]:
                best = (row["id"], sim)
                
        is_dup = best[0] is not None and best[1] >= threshold
        return {
            "is_duplicate": is_dup,
            "duplicate_of": best[0] if is_dup else None,
            "similarity": round(best[1], 4),
            "neighbours": neighbours,
        }
    else:
        # Fallback if using SQLite
        return {
            "is_duplicate": False,
            "duplicate_of": None,
            "similarity": 0.0,
            "neighbours": [],
        }


def reset_index() -> None:
    """Wipe the vector store (used when re-seeding)."""
    if db.IS_POSTGRES:
        with db.connect() as c:
            db._exec(c, "UPDATE complaints SET embedding = NULL")


if __name__ == "__main__":
    db.init_db()
    rows = db.list_complaints()
    print(f"Indexing {len(rows)} complaints...")
    for r in rows:
        index_complaint(r)
        
    # Test: UBI-0001 and UBI-0006 are both UPI failures from Pooja Mishra -- expect a duplicate hit.
    target = next((r for r in rows if r["id"] == "UBI-0006"), None)
    if target:
        result = find_duplicate(target)
        print("Test target:", target["id"], "-", target["customer_name"])
        print("Duplicate?", result["is_duplicate"], "-> of:", result["duplicate_of"],
              "(sim", result["similarity"], ")")
        print("Top neighbours:")
        for n in result["neighbours"][:3]:
            print(f"  {n['id']:10s} sim={n['similarity']}")
