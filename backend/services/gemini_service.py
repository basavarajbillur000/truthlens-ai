
import os
import json
import time
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load the project's .env file reliably
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Streamlit Cloud secrets fallback; local .env takes priority.
if not API_KEY:
    try:
        import streamlit as st
        API_KEY = st.secrets.get("GROQ_API_KEY")
        MODEL = st.secrets.get("GROQ_MODEL", MODEL)
    except Exception:
        API_KEY = None

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Set it in .env or Streamlit Cloud Secrets."
    )

client = Groq(api_key=API_KEY)

# Local embedding model; downloaded on first use
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=4096)
def get_embedding(text):
    model = get_embedding_model()
    vector = model.encode(
        text,
        normalize_embeddings=True
    )
    return tuple(float(value) for value in vector)


def generate_json(prompt, image=None):
    if image is not None:
        raise NotImplementedError(
            "Image analysis is not yet configured for the Groq pipeline."
        )

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful research assistant. "
                            "Return only valid JSON. Do not invent facts "
                            "or evidence."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Groq returned an empty response.")

            return json.loads(content)

        except Exception as error:
            status = getattr(error, "status_code", None)

            if status not in (429, 500, 502, 503, 504):
                raise

            if attempt == max_attempts - 1:
                raise

            time.sleep(2 ** (attempt + 1))


def extract_claims(article):
    prompt = f"""
Analyze the news article below.

Extract independently verifiable factual claims.

IMPORTANT:
- Preserve the exact meaning of the original article.
- Do not change exposure into death, association into causation,
  possibility into certainty, or estimates into confirmed facts.
- Each claim must be traceable to the supplied article.
- Do not add facts that are not present in the article.
- Exclude opinions, predictions, questions and satire.

Return JSON with this structure:
{{
    "topic": "string",
    "claims": ["claim 1", "claim 2"],
    "entities": ["entity 1"],
    "dates": ["date 1"],
    "numbers": ["number 1"],
    "inconsistencies": ["issue 1"]
}}

ARTICLE:
{article}
"""
    return generate_json(prompt)


def analyze_image(image, article_context=""):
    # Image analysis is intentionally disabled until a vision
    # model is configured. Never pretend text-only analysis
    # establishes image authenticity.
    return {
        "description": "Image analysis is not configured yet.",
        "context_match": "inconclusive",
        "visual_observations": [],
        "limitations": [
            "A vision-capable model has not yet been integrated.",
            "No forensic deepfake detection has been performed."
        ]
    }


def verify_claim(claim, evidence):
    if not evidence:
        return {
            "verdict": "inconclusive",
            "explanation": "No relevant evidence was retrieved.",
            "evidence_ids": [],
            "confidence": 0
        }

    evidence_text = "\n".join(
        f"""
Evidence ID: {item['id']}
Source: {item['source']}
Title: {item['title']}
Content: {item['content']}
"""
        for item in evidence
    )

    prompt = f"""
You are a strict, evidence-based fact verification system.

Your task is to determine whether the supplied evidence
actually establishes the claim.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{evidence_text}

Return valid JSON:
{{
    "verdict": "supported, contradicted, mixed, or inconclusive",
    "explanation": "string",
    "evidence_ids": ["exact evidence ID"],
    "confidence": 0
}}

STRICT VERIFICATION RULES:

1. Use ONLY the supplied evidence.
2. Semantic similarity does not establish factual support.
3. Do not use outside knowledge or assumptions.
4. Every important part of the claim must be supported
   for the verdict to be supported.
5. If evidence supports only part of a claim, use mixed.
6. If evidence is relevant but insufficient, use inconclusive.
7. If evidence directly conflicts with the claim, use contradicted.
8. Do not infer rankings, comparisons, causation or superlatives
   unless the evidence explicitly establishes them.
9. Do not turn a related statistic into proof of a broader claim.
10. Cite only evidence IDs supplied above.
11. Confidence must be an integer between 0 and 100.
12. Explain exactly what the evidence establishes and what
    remains unverified.
13. Do not treat evidence supporting the opposite general
    position as proof of a specific claim being false.

14. A claim can be marked contradicted only when the supplied
    evidence directly conflicts with its factual assertion.

15. Absence of supporting evidence is not evidence of falsity.
    If the evidence does not establish or directly contradict
    the claim, use inconclusive.
Be conservative. When uncertain, prefer inconclusive.
"""

    result = generate_json(prompt)

    allowed_verdicts = {
        "supported",
        "contradicted",
        "mixed",
        "inconclusive"
    }

    if result.get("verdict") not in allowed_verdicts:
        raise ValueError("Invalid verification verdict returned.")

    allowed_ids = {item["id"] for item in evidence}

    result["evidence_ids"] = [
        evidence_id
        for evidence_id in result.get("evidence_ids", [])
        if evidence_id in allowed_ids
    ]

    try:
        result["confidence"] = max(
            0, min(100, int(result.get("confidence", 0)))
        )
    except (ValueError, TypeError):
        result["confidence"] = 0

    return result