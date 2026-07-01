"""Classical and retrieval baseline classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines.features import FIX_KEYWORDS, diff_stats, hybrid_feature_vector
from knowledge_base import KnowledgeBase


class BaselineClassifier(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def fit(self, train_rows: list[dict], kb: KnowledgeBase | None = None):
        ...

    @abstractmethod
    def predict(self, commit: dict, kb: KnowledgeBase | None = None) -> dict:
        ...


def _message_diff_text(row: dict) -> str:
    diff = row.get("git_diff", "")
    stats = diff_stats(diff)
    summary = (
        f"files={int(stats['n_files'])} +{int(stats['additions'])} -{int(stats['deletions'])}"
    )
    return f"{row['masked_commit_message']}\nDIFF {summary}\n{diff[:4000]}"


def _lr_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)


def _rf_classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _xgb_classifier():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def _make_tfidf_pipeline(
    classifier,
    *,
    max_features: int = 5000,
) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)),
        ("clf", classifier),
    ])


def _maybe_set_xgb_scale(model: Pipeline, y: list[int]) -> None:
    clf = model.named_steps["clf"]
    if clf.__class__.__name__ != "XGBClassifier":
        return
    pos = sum(1 for v in y if v == 1)
    neg = len(y) - pos
    if pos:
        clf.set_params(scale_pos_weight=neg / pos)


class _TfidfMessageBaseline(BaselineClassifier):
    """Shared TF-IDF-on-message baseline with a pluggable classifier."""

    def __init__(self, *, name: str, description: str, classifier_factory: Callable):
        self.name = name
        self.description = description
        self.model = _make_tfidf_pipeline(classifier_factory(), max_features=5000)

    def fit(self, train_rows, kb=None):
        X = [r["masked_commit_message"] for r in train_rows]
        y = [r["human_label"] for r in train_rows]
        _maybe_set_xgb_scale(self.model, y)
        self.model.fit(X, y)
        return self

    def predict(self, commit, kb=None) -> dict:
        msg = commit["masked_commit_message"]
        pred = int(self.model.predict([msg])[0])
        proba = self.model.predict_proba([msg])[0]
        return {
            "predicted_label": pred,
            "confidence": int(round(max(proba) * 100)),
            "reasoning": f"{self.name} on message (P_bug={proba[1]:.3f})",
        }


class _TfidfMessageDiffBaseline(BaselineClassifier):
    """Shared TF-IDF-on-message+diff baseline with a pluggable classifier."""

    def __init__(self, *, name: str, description: str, classifier_factory: Callable):
        self.name = name
        self.description = description
        self.model = _make_tfidf_pipeline(classifier_factory(), max_features=8000)

    def fit(self, train_rows, kb=None):
        X = [_message_diff_text(r) for r in train_rows]
        y = [r["human_label"] for r in train_rows]
        _maybe_set_xgb_scale(self.model, y)
        self.model.fit(X, y)
        return self

    def predict(self, commit, kb=None) -> dict:
        text = _message_diff_text(commit)
        pred = int(self.model.predict([text])[0])
        proba = self.model.predict_proba([text])[0]
        return {
            "predicted_label": pred,
            "confidence": int(round(max(proba) * 100)),
            "reasoning": f"{self.name} on message+diff (P_bug={proba[1]:.3f})",
        }


class KeywordBaseline(BaselineClassifier):
    name = "keyword"
    description = "Rule: Bug-Fix if message matches fix/bug/patch/hotfix keywords"

    def fit(self, train_rows, kb=None):
        return self

    def predict(self, commit, kb=None) -> dict:
        msg = commit.get("masked_commit_message", "")
        pred = 1 if FIX_KEYWORDS.search(msg) else 0
        return {
            "predicted_label": pred,
            "confidence": 80 if pred else 70,
            "reasoning": f"Keyword heuristic: fix_keywords={'yes' if pred else 'no'}",
        }


class TfidfMessageLR(_TfidfMessageBaseline):
    name = "tfidf_message_lr"
    description = "TF-IDF on masked message → Logistic Regression"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_lr_classifier,
        )


class TfidfMessageRF(_TfidfMessageBaseline):
    name = "tfidf_message_rf"
    description = "TF-IDF on masked message → Random Forest"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_rf_classifier,
        )


class TfidfMessageXGB(_TfidfMessageBaseline):
    name = "tfidf_message_xgb"
    description = "TF-IDF on masked message → XGBoost"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_xgb_classifier,
        )


class TfidfMessageDiffLR(_TfidfMessageDiffBaseline):
    name = "tfidf_message_diff_lr"
    description = "TF-IDF on message + diff summary → Logistic Regression"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_lr_classifier,
        )


class TfidfMessageDiffRF(_TfidfMessageDiffBaseline):
    name = "tfidf_message_diff_rf"
    description = "TF-IDF on message + diff summary → Random Forest"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_rf_classifier,
        )


class TfidfMessageDiffXGB(_TfidfMessageDiffBaseline):
    name = "tfidf_message_diff_xgb"
    description = "TF-IDF on message + diff summary → XGBoost"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_xgb_classifier,
        )


class _HybridFeaturesBaseline(BaselineClassifier):
    """Shared hand-crafted feature baseline with a pluggable classifier."""

    FEATURE_NAMES = [
        "msg_len", "msg_words", "fix_kw", "maint_kw",
        "n_files", "additions", "deletions", "total_changed", "test_ratio",
    ]

    def __init__(self, *, name: str, description: str, classifier_factory: Callable):
        self.name = name
        self.description = description
        self.scaler = StandardScaler()
        self.clf = classifier_factory()

    def fit(self, train_rows, kb=None):
        X = np.array([self._vec(r) for r in train_rows])
        y = [r["human_label"] for r in train_rows]
        if self.clf.__class__.__name__ == "XGBClassifier":
            pos = sum(1 for v in y if v == 1)
            neg = len(y) - pos
            if pos:
                self.clf.set_params(scale_pos_weight=neg / pos)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def _vec(self, row: dict) -> list[float]:
        feats = hybrid_feature_vector(row["masked_commit_message"], row.get("git_diff", ""))
        return [feats[k] for k in self.FEATURE_NAMES]

    def predict(self, commit, kb=None) -> dict:
        X = self.scaler.transform([self._vec(commit)])
        pred = int(self.clf.predict(X)[0])
        proba = self.clf.predict_proba(X)[0]
        return {
            "predicted_label": pred,
            "confidence": int(round(max(proba) * 100)),
            "reasoning": f"{self.name} (P_bug={proba[1]:.3f})",
        }


class HybridFeaturesLR(_HybridFeaturesBaseline):
    name = "hybrid_features_lr"
    description = "Hand-crafted message + diff stats → Logistic Regression"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_lr_classifier,
        )


class HybridFeaturesRF(_HybridFeaturesBaseline):
    name = "hybrid_features_rf"
    description = "Hand-crafted message + diff stats → Random Forest"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_rf_classifier,
        )


class HybridFeaturesXGB(_HybridFeaturesBaseline):
    name = "hybrid_features_xgb"
    description = "Hand-crafted message + diff stats → XGBoost"

    def __init__(self):
        super().__init__(
            name=self.name,
            description=self.description,
            classifier_factory=_xgb_classifier,
        )


class KNNRetrievalBaseline(BaselineClassifier):
    name = "knn_retrieval"
    description = "Label = top-1 KB neighbor (embedding similarity, no LLM)"

    def fit(self, train_rows, kb=None):
        if kb is None:
            raise ValueError("knn_retrieval requires a KnowledgeBase")
        self.kb = kb
        return self

    def predict(self, commit, kb=None) -> dict:
        kb = kb or self.kb
        results = kb.search(
            commit["masked_commit_message"],
            commit.get("git_diff", ""),
            k=1,
            exclude_sha=commit["commit_sha"],
        )
        if not results:
            return {"predicted_label": 0, "confidence": 50, "reasoning": "No KB neighbor found"}
        ref = results[0]
        pred = int(ref["human_label"])
        sim = ref["similarity_score"]
        return {
            "predicted_label": pred,
            "confidence": int(round(sim * 100)),
            "reasoning": (
                f"kNN: nearest neighbor '{ref['message'][:60]}' "
                f"labeled {ref['label_str']} (sim={sim:.3f})"
            ),
            "kb_reference_sha": ref["commit_sha"],
            "kb_similarity": sim,
        }


CLASSICAL_BASELINES = [
    KeywordBaseline,
    TfidfMessageLR,
    TfidfMessageRF,
    TfidfMessageXGB,
    TfidfMessageDiffLR,
    TfidfMessageDiffRF,
    TfidfMessageDiffXGB,
    HybridFeaturesLR,
    HybridFeaturesRF,
    HybridFeaturesXGB,
    KNNRetrievalBaseline,
]
