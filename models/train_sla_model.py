"""Improved SLA breach predictor.

Upgrades vs the previous version:
  * Extended feature engineering (hours_since_filed, pct_sla_elapsed, repeat
    customer, sentiment/severity scores, complaint_word_count, channel_risk,
    is_weekend_filed, is_high_value, has_amount, days_to_sla).
  * **Realistic labels** derived from actual status + filing date + SLA window:
        breach = True   if (status is open  AND days_since_filed > sla_days)
        breach = True   if (status is open  AND severity=Critical
                            AND days_since_filed > sla_days * 0.8)
        breach = True   if (resolved later than sla_days after filing)
        breach = False  otherwise
  * 5-fold stratified CV across:
        - Random Forest (class_weight balanced)
        - XGBoost (small GridSearchCV over a sane sweep)
        - LightGBM
        - Stacking ensemble (RF + XGB + LGBM) with Logistic Regression meta-learner
  * SMOTE inside the imblearn Pipeline (so resampling only touches training folds)
  * Saves the winner as BOTH `models/sla_best_model.joblib` AND
    `models/sla_rf.joblib` (for backward-compat with `sla_monitor.py`).
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import warnings
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import OneHotEncoder

# Conditional imports for imblearn and lightgbm to prevent training failure on systems without them.
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = False

from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "complaints.json"
SLA_RULES = ROOT / "data" / "sla_rules.json"
MODEL_OUT_LEGACY = ROOT / "models" / "sla_rf.joblib"      # consumed by sla_monitor.py
MODEL_OUT_BEST   = ROOT / "models" / "sla_best_model.joblib"
META_OUT         = ROOT / "models" / "sla_leaderboard.json"

CATEGORICAL = ["channel", "language", "account_type", "category", "severity",
               "sentiment"]
NUMERIC = [
    "amount_involved", "complaint_text_length", "complaint_word_count",
    "hours_since_filed", "day_of_week", "is_weekend_filed",
    "is_high_amount", "is_high_value", "is_fraud_keyword",
    "is_duplicate", "is_repeat_customer", "has_amount",
    "channel_risk", "sentiment_score", "severity_score",
    "customer_complaint_count", "days_to_sla", "pct_sla_elapsed",
]

FRAUD_KEYWORDS = ("unauthor", "fraud", "stolen", "hack", "scam", "stuck", "debited")
CATEGORY_KEYWORDS = [
    ("UPI",        ("upi",)),
    ("ATM",        ("atm",)),
    ("Card",       ("credit card", "debit card", "card")),
    ("Loan",       ("loan", "emi")),
    ("NetBanking", ("net banking", "mobile banking", "internet banking", "netbanking")),
]
SEVERITY_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
SENTIMENT_ORDER = {"Angry": 4, "Frustrated": 3, "Neutral": 2, "Polite": 1}
HIGH_VISIBILITY_CHANNELS = {"twitter", "whatsapp"}

TODAY = date(2026, 5, 29)


def _load_sla_rules() -> dict:
    return json.loads(SLA_RULES.read_text(encoding="utf-8"))


# --- feature engineering ------------------------------------------------------

def engineer_features(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    today_ts = pd.Timestamp(today or TODAY).tz_localize(None)
    df = df.copy()
    df["complaint_text"] = df["complaint_text"].fillna("")
    df["amount_involved"] = df["amount_involved"].fillna(0).astype(float)
    df["complaint_text_length"] = df["complaint_text"].str.len().astype(int)
    df["complaint_word_count"] = (df["complaint_text"]
                                  .str.split().str.len().fillna(0).astype(int))

    dates = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df["day_of_week"] = dates.dt.weekday.fillna(0).astype(int)
    df["is_weekend_filed"] = (df["day_of_week"] >= 5).astype(int)
    df["is_high_amount"] = (df["amount_involved"] >= 25000).astype(int)
    df["is_high_value"] = (df["amount_involved"] > 50000).astype(int)
    df["has_amount"] = (df["amount_involved"] > 0).astype(int)
    df["is_fraud_keyword"] = df["complaint_text"].str.lower().apply(
        lambda t: int(any(k in t for k in FRAUD_KEYWORDS))
    )

    hours_since = (today_ts - dates).dt.total_seconds() / 3600.0
    df["hours_since_filed"] = hours_since.fillna(0).astype(float)

    # Repeat-customer + complaint count.
    counts = df.groupby("customer_name").size().rename("customer_complaint_count")
    df = df.merge(counts, on="customer_name", how="left")
    df["is_repeat_customer"] = (df["customer_complaint_count"] > 1).astype(int)

    df["channel_risk"] = df["channel"].fillna("").isin(HIGH_VISIBILITY_CHANNELS).astype(int)

    # Backfill LLM-only fields with safe heuristics so the feature set is always defined.
    df["category"]  = df["category"].fillna(
        df["complaint_text"].apply(_heuristic_category))
    df["severity"]  = df["severity"].fillna("Medium")
    df["sentiment"] = df["sentiment"].fillna("Neutral")

    df["severity_score"]  = df["severity"].map(SEVERITY_ORDER).fillna(2).astype(int)
    df["sentiment_score"] = df["sentiment"].map(SENTIMENT_ORDER).fillna(2).astype(int)

    # SLA days from rules.json, adjusted by severity multiplier.
    rules = _load_sla_rules()
    base_sla = df["category"].map(rules["sla_days"]).fillna(rules["sla_days"]["General"])
    sev_mult = df["severity"].map(rules["severity_multiplier"]).fillna(1.0)
    df["days_to_sla"] = (base_sla * sev_mult).clip(lower=1).astype(int)
    total_sla_hours = df["days_to_sla"] * 24.0
    df["pct_sla_elapsed"] = (df["hours_since_filed"] / total_sla_hours).fillna(0.0)

    if "is_duplicate" not in df.columns or df["is_duplicate"].isna().all():
        df["is_duplicate"] = (df.get("duplicate_of").notna().astype(int)
                              if "duplicate_of" in df.columns else 0)
    df["is_duplicate"] = df["is_duplicate"].fillna(0).astype(int)
    return df


def _heuristic_category(text: str) -> str:
    t = (text or "").lower()
    for cat, kws in CATEGORY_KEYWORDS:
        if any(k in t for k in kws):
            return cat
    return "General"


# --- realistic label generation ----------------------------------------------

def realistic_labels(df: pd.DataFrame, today: date | None = None) -> np.ndarray:
    """breach = 1 if:
        - status is open AND days_since_filed > days_to_sla
        - status is open AND severity Critical AND days_since_filed > 0.8 * sla
        - status is resolved/auto-resolved AND resolved_at - date > sla_days
       Otherwise 0.
    """
    today_ts = pd.Timestamp(today or TODAY).tz_localize(None)
    dates = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    days_since = (today_ts - dates).dt.days.fillna(0)
    status = df.get("status", pd.Series(["open"] * len(df))).fillna("open")
    resolved_at = pd.to_datetime(df.get("resolved_at"), errors="coerce").dt.tz_localize(None)
    resolution_days = (resolved_at - dates).dt.days

    sla = df["days_to_sla"].astype(int)
    severity = df.get("severity", pd.Series(["Medium"] * len(df)))

    is_open = ~status.isin(("resolved", "auto_resolved_dup", "auto_resolved_std"))
    breach = pd.Series(0, index=df.index)

    # Open + past due
    breach = breach.mask(is_open & (days_since > sla), 1)
    # Open + Critical + within 80% of SLA already used up
    breach = breach.mask(is_open & (severity == "Critical")
                         & (days_since > 0.8 * sla), 1)
    # Resolved past SLA
    breach = breach.mask((~is_open) & (resolution_days > sla), 1)
    return breach.astype(int).to_numpy()


# --- data loaders -------------------------------------------------------------

def load_training_data() -> pd.DataFrame:
    """Use the LLM-labelled DB rows. Falls back to JSON only when DB empty."""
    from database import db
    db.init_db()
    rows = db.list_complaints(where="processed_at IS NOT NULL")
    if len(rows) >= 100:
        return pd.DataFrame(rows)
    df = pd.DataFrame(json.loads(DATA.read_text(encoding="utf-8")))
    for c in ("category", "severity", "sentiment", "duplicate_of",
              "status", "resolved_at"):
        if c not in df:
            df[c] = pd.NA
    return df


# --- model factory ------------------------------------------------------------

def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ],
        remainder="drop",
    )


def _make_pipeline(estimator, *, use_smote: bool = True) -> ImbPipeline:
    steps = [("pre", _preprocessor())]
    if use_smote and HAS_IMBLEARN:
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=5)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def build_candidates() -> dict[str, ImbPipeline]:
    candidates = {
        "RandomForest": _make_pipeline(RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
        "GradientBoosting": _make_pipeline(GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.06, random_state=42,
        )),
        "XGBoost": _make_pipeline(XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", n_jobs=-1, random_state=42,
        )),
    }
    if HAS_LIGHTGBM:
        candidates["LightGBM"] = _make_pipeline(LGBMClassifier(
            n_estimators=500, num_leaves=31, max_depth=-1,
            learning_rate=0.05, min_child_samples=10, subsample=0.9,
            colsample_bytree=0.9, random_state=42, n_jobs=-1, verbose=-1,
        ))
    return candidates


def build_stacking() -> ImbPipeline:
    base = [
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", n_jobs=-1, random_state=42)),
    ]
    if HAS_LIGHTGBM:
        base.append(("lgb", LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbose=-1)))
            
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    return _make_pipeline(StackingClassifier(
        estimators=base, final_estimator=meta,
        cv=5, n_jobs=-1, passthrough=False))


# --- training driver ---------------------------------------------------------

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_model(name: str, pipe: ImbPipeline, X, y) -> dict:
    aucs = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc", n_jobs=1)
    accs = cross_val_score(pipe, X, y, cv=CV, scoring="accuracy", n_jobs=1)
    return {
        "model": name,
        "cv_auc_mean": round(float(aucs.mean()), 4),
        "cv_auc_std":  round(float(aucs.std()), 4),
        "cv_acc_mean": round(float(accs.mean()), 4),
        "fold_aucs":   [round(float(a), 4) for a in aucs],
    }


def tune_xgboost(X, y) -> tuple[ImbPipeline, dict]:
    """Light GridSearchCV on a sensible XGBoost sweep (no SMOTE inside the
    grid -- we apply SMOTE in the final fit)."""
    base = _make_pipeline(XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", n_jobs=-1, random_state=42,
        subsample=0.9, colsample_bytree=0.9,
    ), use_smote=False)
    param_grid = {
        "clf__n_estimators":     [200, 400],
        "clf__max_depth":        [3, 5, 7],
        "clf__learning_rate":    [0.05, 0.1],
        "clf__min_child_weight": [1, 3],
    }
    gs = GridSearchCV(
        base, param_grid, cv=3, scoring="roc_auc",
        n_jobs=-1, refit=False, verbose=0,
    )
    gs.fit(X, y)
    best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    print(f"  XGBoost grid best AUC={gs.best_score_:.4f}  params={best_params}")
    tuned = _make_pipeline(XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", n_jobs=-1, random_state=42,
        subsample=0.9, colsample_bytree=0.9,
        **best_params,
    ))
    return tuned, best_params


def main() -> None:
    df = load_training_data()
    df = engineer_features(df)
    y = realistic_labels(df)
    print(f"Training rows: {len(df)} | breach rate: {y.mean():.2%}")

    X = df[CATEGORICAL + NUMERIC]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y if y.sum() else None,
    )

    candidates = build_candidates()

    # ---- XGBoost: GridSearch first, then plug the tuned version back into the
    #      candidates dict so it gets evaluated like everyone else.
    print("\n[GridSearchCV] tuning XGBoost...")
    tuned_xgb, xgb_best = tune_xgboost(X_tr, y_tr)
    candidates["XGBoost(tuned)"] = tuned_xgb

    print(f"\n[5-fold StratifiedCV] evaluating {len(candidates) + 1} models...")
    leaderboard: list[dict] = []
    for name, pipe in candidates.items():
        res = evaluate_model(name, pipe, X_tr, y_tr)
        leaderboard.append(res)
        print(f"  {name:18s} AUC={res['cv_auc_mean']:.4f} +/- {res['cv_auc_std']:.4f}  "
              f"ACC={res['cv_acc_mean']:.4f}")

    # ---- Stacking ensemble ----
    print("\n[Stacking] RF + XGB + LGBM with LogReg meta-learner...")
    stack = build_stacking()
    stack_res = evaluate_model("Stacking", stack, X_tr, y_tr)
    leaderboard.append(stack_res)
    print(f"  Stacking           AUC={stack_res['cv_auc_mean']:.4f} +/- "
          f"{stack_res['cv_auc_std']:.4f}  ACC={stack_res['cv_acc_mean']:.4f}")

    # ---- Pick the winner by CV AUC ----
    leaderboard.sort(key=lambda r: -r["cv_auc_mean"])
    winner_name = leaderboard[0]["model"]
    winner_pipe = (stack if winner_name == "Stacking" else candidates[winner_name])
    print(f"\nWINNER: {winner_name} (CV AUC = {leaderboard[0]['cv_auc_mean']:.4f})")

    # Fit the winner on the FULL training fold and evaluate on the held-out 20%.
    winner_pipe.fit(X_tr, y_tr)
    prob = winner_pipe.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    holdout_auc = float(roc_auc_score(y_te, prob))
    holdout_acc = float(accuracy_score(y_te, pred))
    print(f"\nHold-out AUC: {holdout_auc:.4f}  ACC: {holdout_acc:.4f}")
    print("\nHold-out report:")
    print(classification_report(y_te, pred, digits=3, zero_division=0))

    # Feature importances when the underlying estimator exposes them.
    importances = _extract_importances(winner_pipe)

    MODEL_OUT_LEGACY.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": winner_pipe,
        "categorical": CATEGORICAL,
        "numeric": NUMERIC,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "winner": winner_name,
        "leaderboard": leaderboard,
        "feature_importances": importances,
        "training_rows": len(df),
        "test_metrics": {
            "cv_auc": leaderboard[0]["cv_auc_mean"],
            "cv_auc_std": leaderboard[0]["cv_auc_std"],
            "holdout_auc": round(holdout_auc, 4),
            "holdout_accuracy": round(holdout_acc, 4),
        },
        "xgb_best_params": xgb_best,
        "feature_helpers": {
            "fraud_keywords": FRAUD_KEYWORDS,
            "category_keywords": CATEGORY_KEYWORDS,
            "severity_order": SEVERITY_ORDER,
            "sentiment_order": SENTIMENT_ORDER,
            "high_visibility_channels": list(HIGH_VISIBILITY_CHANNELS),
        },
    }
    joblib.dump(payload, MODEL_OUT_BEST)
    joblib.dump(payload, MODEL_OUT_LEGACY)
    META_OUT.write_text(json.dumps({
        "winner": winner_name,
        "leaderboard": leaderboard,
        "feature_importances": importances,
        "training_rows": len(df),
        "trained_at": payload["trained_at"],
        "holdout_auc": payload["test_metrics"]["holdout_auc"],
        "holdout_accuracy": payload["test_metrics"]["holdout_accuracy"],
        "xgb_best_params": xgb_best,
    }, indent=2))
    print(f"\nSaved artefact -> {MODEL_OUT_BEST}")
    print(f"               -> {MODEL_OUT_LEGACY}  (legacy filename)")
    print(f"Leaderboard    -> {META_OUT}")


def _extract_importances(pipe: ImbPipeline) -> dict[str, float]:
    """Return {feature_name: importance} for tree estimators. Stacking returns
    the mean importance across the base estimators."""
    pre = pipe.named_steps["pre"]
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = []
    clf = pipe.named_steps["clf"]

    def _vec(est) -> np.ndarray | None:
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=float)
        return None

    if isinstance(clf, StackingClassifier):
        vecs = [v for _, est in clf.named_estimators_.items() if (v := _vec(est)) is not None]
        if not vecs:
            return {}
        # All vecs have the same length (= # transformed features); mean them.
        imp = np.mean(np.stack(vecs), axis=0)
    else:
        imp = _vec(clf)
        if imp is None:
            return {}
    if not names or len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]
    pairs = sorted(zip(names, imp.tolist()), key=lambda p: -p[1])
    return {n: round(float(v), 5) for n, v in pairs[:30]}


if __name__ == "__main__":
    main()
