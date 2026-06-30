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
_collection = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from pathlib import Path
        db_path = Path(__file__).resolve().parent.parent / "data" / "chroma"
        client = chromadb.PersistentClient(path=str(db_path))
        _collection = client.get_or_create_collection(
            name="complaints",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


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
    else:
        coll = _get_collection()
        coll.upsert(
            ids=[cid],
            embeddings=[vec],
            metadatas=[{
                "customer_name": complaint.get("customer_name", ""),
                "channel": complaint.get("channel", ""),
                "date": complaint.get("date", ""),
                "category": complaint.get("category", ""),
                "location": complaint.get("location", ""),
            }]
        )


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
        # Fallback if using SQLite (ChromaDB)
        coll = _get_collection()
        res = coll.query(
            query_embeddings=[qvec],
            n_results=top_k + 1,  # +1 because the query itself might be returned
            where={"customer_name": customer} if customer else None
        )
        
        neighbours = []
        best = (None, 0.0)
        
        if res and res["ids"] and len(res["ids"][0]) > 0:
            for i in range(len(res["ids"][0])):
                nid = res["ids"][0][i]
                if nid == cid:
                    continue  # skip self
                
                # ChromaDB cosine distance: similarity = 1 - distance
                sim = 1.0 - (res["distances"][0][i] if res["distances"] else 0.0)
                meta = res["metadatas"][0][i] if res["metadatas"] else {}
                
                neighbours.append({
                    "id": nid,
                    "similarity": round(sim, 4),
                    "meta": meta
                })
                
                if sim > best[1]:
                    best = (nid, sim)
                    
        is_dup = best[0] is not None and best[1] >= threshold
        return {
            "is_duplicate": is_dup,
            "duplicate_of": best[0] if is_dup else None,
            "similarity": round(best[1], 4),
            "neighbours": neighbours[:top_k],
        }


def reset_index() -> None:
    """Wipe the vector store (used when re-seeding)."""
    if db.IS_POSTGRES:
        with db.connect() as c:
            db._exec(c, "UPDATE complaints SET embedding = NULL")
    else:
        import chromadb
        from pathlib import Path
        db_path = Path(__file__).resolve().parent.parent / "data" / "chroma"
        client = chromadb.PersistentClient(path=str(db_path))
        try:
            client.delete_collection("complaints")
        except ValueError:
            pass
        global _collection
        _collection = None


def get_all_embeddings() -> tuple[list[str], list[list[float]], list[dict[str, Any]]]:
    """Fetch all embeddings for clustering (used by Root Cause agent)."""
    if db.IS_POSTGRES:
        query = "SELECT id, category, location, embedding FROM complaints WHERE embedding IS NOT NULL"
        with db.connect() as c:
            rows = db._exec(c, query).fetchall()
        
        ids = []
        vecs = []
        metas = []
        for r in rows:
            ids.append(r["id"])
            emb = r["embedding"]
            if isinstance(emb, str):
                import ast
                emb = ast.literal_eval(emb)
            elif hasattr(emb, "tolist"):
                emb = emb.tolist()
            vecs.append(emb)
            metas.append({
                "category": r["category"],
                "location": r["location"]
            })
        return ids, vecs, metas
    else:
        coll = _get_collection()
        res = coll.get(include=["embeddings", "metadatas"])
        ids = list(res.get("ids") or [])
        embs = list(res.get("embeddings") or [])
        metas = list(res.get("metadatas") or [])
        return ids, embs, metas


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
