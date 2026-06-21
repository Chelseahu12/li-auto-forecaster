"""Aspect-level sentiment scoring using Chinese RoBERTa.

Model: hfl/chinese-roberta-wwm-ext
Zero-shot approach: compare review embeddings to positive/negative anchor phrases
via cosine similarity, then normalize to [0, 1] per aspect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASPECTS = ["range", "interior", "price_value", "performance", "aftersales"]

_POSITIVE_ANCHORS = {
    "range": "续航里程很长，完全没有里程焦虑",
    "interior": "内饰豪华精致，做工细腻",
    "price_value": "价格合理，性价比很高",
    "performance": "加速强劲，驾驶感受出色",
    "aftersales": "售后服务非常好，响应及时",
}
_NEGATIVE_ANCHORS = {
    "range": "续航很短，里程焦虑严重",
    "interior": "内饰简陋，做工差",
    "price_value": "价格太贵，性价比低",
    "performance": "加速慢，驾驶体验差",
    "aftersales": "售后服务很差，无人理睬",
}

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        _model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        _model.eval()
    return _tokenizer, _model


def _embed(texts: list[str]) -> np.ndarray:
    import torch
    tokenizer, model = _load_model()
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state.mean(dim=1).numpy()


def score_reviews(texts: list[str]) -> list[dict[str, float]]:
    """Score each review text on all aspects. Returns list of {aspect: score} dicts."""
    all_texts = texts + list(_POSITIVE_ANCHORS.values()) + list(_NEGATIVE_ANCHORS.values())
    embeddings = _embed(all_texts)

    n = len(texts)
    review_embs = embeddings[:n]
    pos_embs = embeddings[n: n + len(ASPECTS)]
    neg_embs = embeddings[n + len(ASPECTS):]

    def _cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    scores = []
    for emb in review_embs:
        row = {}
        for i, aspect in enumerate(ASPECTS):
            pos_sim = _cosine(emb, pos_embs[i])
            neg_sim = _cosine(emb, neg_embs[i])
            row[aspect] = (pos_sim + 1) / (pos_sim + neg_sim + 2)
        scores.append(row)
    return scores


_PRIOR = {a: 0.6 for a in ASPECTS}
_PRIOR_WEIGHT = 3.0


def aggregate_sentiment(
    reviews_df: pd.DataFrame,
    feature_months: list[int] = None,
    prior: dict[str, float] = None,
) -> pd.DataFrame:
    """Score all reviews in feature_months and aggregate to one row per model."""
    if feature_months is None:
        feature_months = [1, 2, 3]
    if prior is None:
        prior = _PRIOR

    early = reviews_df[reviews_df["month_since_launch"].isin(feature_months)].copy()

    rows = []
    for model, group in early.groupby("model"):
        texts = group["review_text"].tolist()
        if not texts:
            row = {"model": model, **{f"sent_{a}": prior[a] for a in ASPECTS}}
        else:
            scored = score_reviews(texts)
            row = {"model": model}
            for aspect in ASPECTS:
                raw_scores = [s[aspect] for s in scored]
                smoothed = (sum(raw_scores) + _PRIOR_WEIGHT * prior[aspect]) / (
                    len(raw_scores) + _PRIOR_WEIGHT
                )
                row[f"sent_{aspect}"] = smoothed
        rows.append(row)
    return pd.DataFrame(rows)
