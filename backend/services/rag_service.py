import json
import numpy as np

from pathlib import Path
from backend.services.gemini_service import get_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "evidence.json"


def load_knowledge_base():
    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KB_PATH}"
        )

    with open(KB_PATH, "r", encoding="utf-8") as file:
        knowledge_base = json.load(file)

    if not isinstance(knowledge_base, list):
        raise ValueError("Knowledge base must contain a JSON list.")

    return knowledge_base


def retrieve_evidence(claim, knowledge_base, top_k=3):
    if not claim.strip():
        return []

    claim_vector = np.array(
        get_embedding(claim),
        dtype=np.float32
    )

    results = []

    for item in knowledge_base:
        evidence_vector = np.array(
            get_embedding(item["content"]),
            dtype=np.float32
        )

        denominator = (
            np.linalg.norm(claim_vector)
            * np.linalg.norm(evidence_vector)
        )

        similarity = (
            float(
                np.dot(claim_vector, evidence_vector)
                / denominator
            )
            if denominator else 0.0
        )

        results.append({
            **item,
            "similarity": similarity
        })

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:top_k]