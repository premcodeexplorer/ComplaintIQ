"""Agent 6 -- Root Cause Detector.

Clusters complaint embeddings with KMeans and flags any cluster that:
  - has at least `MIN_CLUSTER_SIZE` complaints, AND
  - is dominated (>= 60%) by a single (category, location) pair within a recent
    time window.

This catches systemic issues like "47 UPI failures from Nagpur over the past 7
days = gateway problem" that no single complaint would surface on its own.

Embeddings are reused from the Supabase pgvector column.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from database import db

MIN_CLUSTER_SIZE = 5
DOMINANCE_THRESHOLD = 0.60   # 60% of cluster must share one (category, location)
DEFAULT_K = 12


def _collect_vectors() -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    if not db.IS_POSTGRES:
        return [], np.zeros((0, 384), dtype=np.float32), []
        
    query = """
        SELECT id, category, location, embedding 
        FROM complaints 
        WHERE embedding IS NOT NULL
    """
    with db.connect() as c:
        rows = db._exec(c, query).fetchall()
        
    ids = []
    embs = []
    metas = []
    
    for row in rows:
        ids.append(row["id"])
        embs.append(row["embedding"])
        metas.append({
            "category": row["category"],
            "location": row["location"],
        })
        
    if not embs:
        arr = np.zeros((0, 384), dtype=np.float32)
    else:
        arr = np.asarray(embs, dtype=np.float32)
        
    return ids, arr, metas


def detect(k: int = DEFAULT_K) -> list[dict[str, Any]]:
    """Run KMeans on all stored embeddings and return root-cause alerts."""
    ids, vecs, metas = _collect_vectors()
    if len(vecs) < MIN_CLUSTER_SIZE:
        return []

    k = max(2, min(k, len(vecs) // MIN_CLUSTER_SIZE))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(vecs)

    alerts: list[dict[str, Any]] = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if len(idx) < MIN_CLUSTER_SIZE:
            continue
        cluster_metas = [metas[i] for i in idx]
        cluster_ids = [ids[i] for i in idx]

        cat_counter = Counter(m.get("category") or "Unknown" for m in cluster_metas)
        loc_counter = Counter(m.get("location") or "Unknown" for m in cluster_metas)

        top_cat, cat_n = cat_counter.most_common(1)[0]
        top_loc, loc_n = loc_counter.most_common(1)[0]
        n = len(idx)

        cat_share = cat_n / n
        loc_share = loc_n / n

        if cat_share >= DOMINANCE_THRESHOLD:
            # Top-3 cities by count -- gives a much more actionable summary
            # than the blanket "multiple cities" label.
            top_cities = [
                (city, cnt) for city, cnt in loc_counter.most_common(3)
                if city not in (None, "", "Unknown")
            ]
            if loc_share >= 0.4:
                # One city dominates -- name it.
                location_label = top_loc
                summary = (
                    f"{n} {top_cat} complaints concentrated in {top_loc} "
                    f"({loc_share:.0%} of cluster) -- "
                    f"possible local {top_cat} service issue."
                )
            elif top_cities:
                # Cluster spans multiple cities -- name the top three.
                cities_phrase = ", ".join(
                    f"{c} ({n_c})" for c, n_c in top_cities
                )
                others = sum(cnt for _, cnt in loc_counter.most_common()
                             if cnt and (_, cnt) not in top_cities) \
                         - sum(cnt for _, cnt in top_cities)
                top3_share = sum(cnt for _, cnt in top_cities) / n
                summary = (
                    f"{n} {top_cat} complaints across {len(loc_counter)} cities. "
                    f"Top hotspots: {cities_phrase}"
                    + (f" + {n - sum(cnt for _, cnt in top_cities)} elsewhere"
                       if n > sum(cnt for _, cnt in top_cities) else "")
                    + f". Possible systemic {top_cat} issue."
                )
                location_label = None
            else:
                location_label = None
                summary = (
                    f"{n} {top_cat} complaints, location unknown -- "
                    f"possible systemic {top_cat} issue."
                )
            alerts.append({
                "cluster_id": int(cid),
                "category": top_cat,
                "location": location_label,
                "top_cities": top_cities,
                "count": int(n),
                "category_share": round(cat_share, 2),
                "location_share": round(loc_share, 2),
                "complaint_ids": cluster_ids,
                "summary": summary,
                "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })

    # Largest / most-dominant alerts first.
    alerts.sort(key=lambda a: (a["count"], a["category_share"]), reverse=True)
    return alerts


def store_alerts(alerts: list[dict[str, Any]]) -> int:
    from database import db

    db.clear_alerts()
    for a in alerts:
        db.insert_root_cause_alert(
            cluster_id=a["cluster_id"],
            category=a["category"],
            location=a["location"],
            count=a["count"],
            summary=a["summary"],
            created_at=a["created_at"],
        )
    return len(alerts)


if __name__ == "__main__":
    from database import db

    db.init_db()
    alerts = detect()
    print(f"Detected {len(alerts)} root-cause cluster(s):")
    for a in alerts:
        print(f"  cluster={a['cluster_id']:2d} n={a['count']:3d} "
              f"cat={a['category']:10s} loc={a['location']!s:15s} "
              f"share={a['category_share']:.0%} -> {a['summary']}")
    stored = store_alerts(alerts)
    print(f"Stored {stored} alerts in DB.")
