"""Agent 3 -- Duplicate Detector.

Embeds complaint text with sentence-transformers (all-MiniLM-L6-v2) and stores
embeddings in a persistent ChromaDB collection. For each new complaint, queries
the top-k nearest neighbours from the same customer and returns the best match
above a similarity threshold.

Why "same customer"? Two different customers reporting "UPI failed" is NOT a
duplicate -- it's a systemic issue (which Agent 6 handles). A duplicate means
the *same* customer raised the *same* complaint twice across channels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION = "complaints"
MODEL_NAME = "all-MiniLM-L6-v2"
DUP_THRESHOLD = 0.78  # cosine similarity

_model = None
_chroma_client = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        from chromadb.utils import embedding_functions as _ef

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Pin the embedding function to sentence-transformers so Chroma never
        # falls back to its DEFAULT ONNX embedder. That default needs
        # onnxruntime, whose DLL is fragile under Windows load order and throws
        # a misleading "onnxruntime not installed" error inside Streamlit. We
        # pass embeddings explicitly on every call, so this mainly blocks the
        # ONNX import path. Cosine space -> distance = 1 - cosine_similarity.
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ef.SentenceTransformerEmbeddingFunction(
                model_name=MODEL_NAME
            ),
        )
    return _collection


def embed(text: str) -> list[float]:
    return _get_model().encode([text], normalize_embeddings=True)[0].tolist()


def index_complaint(complaint: dict[str, Any]) -> None:
    """Add a complaint to the vector store. Idempotent on `id`."""
    coll = _get_collection()
    cid = complaint["id"]
    # If already indexed, skip (chroma upsert keeps things simple).
    text = complaint.get("complaint_text", "") or ""
    coll.upsert(
        ids=[cid],
        embeddings=[embed(text)],
        documents=[text],
        metadatas=[{
            "customer_name": complaint.get("customer_name") or "",
            "channel": complaint.get("channel") or "",
            "date": complaint.get("date") or "",
            "category": complaint.get("category") or "",
            "location": complaint.get("location") or "",
        }],
    )


def find_duplicate(complaint: dict[str, Any], *, threshold: float = DUP_THRESHOLD,
                   top_k: int = 5) -> dict[str, Any]:
    """Return {is_duplicate, duplicate_of, similarity, neighbours[]}.

    Restricts the search to the same customer (see module docstring).
    """
    coll = _get_collection()
    text = complaint.get("complaint_text", "") or ""
    qvec = embed(text)

    customer = complaint.get("customer_name") or ""
    # Chroma `where` clause: only same-customer rows, exclude this complaint itself.
    where: dict[str, Any] = {"customer_name": customer} if customer else {}

    res = coll.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where or None,
    )

    ids = (res.get("ids") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    neighbours: list[dict[str, Any]] = []
    best: tuple[str | None, float] = (None, 0.0)
    for nid, dist, meta in zip(ids, distances, metas):
        if nid == complaint.get("id"):
            continue
        sim = max(0.0, 1.0 - float(dist))
        neighbours.append({"id": nid, "similarity": round(sim, 4), "meta": meta})
        if sim > best[1]:
            best = (nid, sim)

    is_dup = best[0] is not None and best[1] >= threshold
    return {
        "is_duplicate": is_dup,
        "duplicate_of": best[0] if is_dup else None,
        "similarity": round(best[1], 4),
        "neighbours": neighbours,
    }


def reset_index() -> None:
    """Wipe the vector store (used when re-seeding)."""
    global _chroma_client, _collection
    coll = _get_collection()
    coll.delete(where={})


if __name__ == "__main__":
    from database import db

    db.init_db()
    rows = db.list_complaints()
    print(f"Indexing {len(rows)} complaints...")
    for r in rows:
        index_complaint(r)
    coll = _get_collection()
    print(f"Chroma now holds {coll.count()} vectors.")

    # Test: UBI-0001 and UBI-0006 are both UPI failures from Pooja Mishra -- expect a duplicate hit.
    target = next(r for r in rows if r["id"] == "UBI-0006")
    result = find_duplicate(target)
    print("Test target:", target["id"], "-", target["customer_name"])
    print("Duplicate?", result["is_duplicate"], "-> of:", result["duplicate_of"],
          "(sim", result["similarity"], ")")
    print("Top neighbours:")
    for n in result["neighbours"][:3]:
        print(f"  {n['id']:10s} sim={n['similarity']}")
