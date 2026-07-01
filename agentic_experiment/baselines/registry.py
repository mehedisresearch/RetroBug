"""Baseline registry."""

from __future__ import annotations

from baselines.classical import (
    CLASSICAL_BASELINES,
    HybridFeaturesLR,
    HybridFeaturesRF,
    HybridFeaturesXGB,
    KeywordBaseline,
    KNNRetrievalBaseline,
    TfidfMessageDiffLR,
    TfidfMessageDiffRF,
    TfidfMessageDiffXGB,
    TfidfMessageLR,
    TfidfMessageRF,
    TfidfMessageXGB,
)
from baselines.llm_baselines import (
    ClaudeSingleShotBaseline,
    CodeLlamaSingleShotBaseline,
    Gpt4oSingleShotBaseline,
    LLM_BASELINES,
)

BASELINE_NAMES = [
    "keyword",
    "tfidf_message_lr",
    "tfidf_message_rf",
    "tfidf_message_xgb",
    "tfidf_message_diff_lr",
    "tfidf_message_diff_rf",
    "tfidf_message_diff_xgb",
    "hybrid_features_lr",
    "hybrid_features_rf",
    "hybrid_features_xgb",
    "knn_retrieval",
    "claude_message",
    "claude_message_diff",
    "gpt4o_message_diff",
    "codellama_message_diff",
]

CLASSICAL_ONLY = [
    "keyword",
    "tfidf_message_lr",
    "tfidf_message_rf",
    "tfidf_message_xgb",
    "tfidf_message_diff_lr",
    "tfidf_message_diff_rf",
    "tfidf_message_diff_xgb",
    "hybrid_features_lr",
    "hybrid_features_rf",
    "hybrid_features_xgb",
    "knn_retrieval",
]

LLM_ONLY = [
    "claude_message",
    "claude_message_diff",
    "gpt4o_message_diff",
    "codellama_message_diff",
]


def get_baseline(name: str):
    mapping = {
        "keyword": KeywordBaseline,
        "tfidf_message_lr": TfidfMessageLR,
        "tfidf_message_rf": TfidfMessageRF,
        "tfidf_message_xgb": TfidfMessageXGB,
        "tfidf_message_diff_lr": TfidfMessageDiffLR,
        "tfidf_message_diff_rf": TfidfMessageDiffRF,
        "tfidf_message_diff_xgb": TfidfMessageDiffXGB,
        "hybrid_features_lr": HybridFeaturesLR,
        "hybrid_features_rf": HybridFeaturesRF,
        "hybrid_features_xgb": HybridFeaturesXGB,
        "knn_retrieval": KNNRetrievalBaseline,
        "claude_message": lambda: ClaudeSingleShotBaseline(message_only=True),
        "claude_message_diff": lambda: ClaudeSingleShotBaseline(message_only=False),
        "gpt4o_message_diff": Gpt4oSingleShotBaseline,
        "codellama_message_diff": CodeLlamaSingleShotBaseline,
    }
    if name not in mapping:
        raise ValueError(f"Unknown baseline: {name}. Choose from {BASELINE_NAMES}")
    factory = mapping[name]
    if isinstance(factory, type):
        return factory()
    return factory()
