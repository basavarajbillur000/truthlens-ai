import streamlit as st
import json
import time

from PIL import Image
from io import BytesIO

from backend.services.gemini_service import (
    extract_claims,
    analyze_image,
    verify_claim
)

from backend.services.rag_service import (
    load_knowledge_base,
    retrieve_evidence
)

from backend.services.live_evidence_service import search_live_sources


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide"
)
# --------------------------------------------------
# TRUTHLENS AI — INVESTIGATIVE DASHBOARD UI
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#080c16;--surface:#101827;--border:rgba(150,167,197,.16);--muted:#93a1ba;--violet:#8b7cff;}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important;height:0!important;visibility:hidden!important;}
.stApp{background:radial-gradient(ellipse at 82% 0%,rgba(111,83,235,.17),transparent 33%),radial-gradient(ellipse at 0% 48%,rgba(31,132,160,.075),transparent 32%),var(--bg);color:#edf2fc;font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"]>.main{background:transparent;}
.block-container{max-width:1380px!important;padding:3.25rem 3.1rem 4.5rem!important;}
[data-testid="stSidebar"]{background:#0d1422;border-right:1px solid var(--border);}
[data-testid="stSidebar"] .block-container{padding:2rem 1.35rem!important;}
h1,h2,h3{color:#f5f7ff!important;font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em;}
h1{font-size:clamp(2.25rem,4vw,3.15rem)!important;font-weight:700!important;}h2{font-size:1.7rem!important;font-weight:650!important;}h3{font-size:1.12rem!important;font-weight:600!important;}
p,label,.stMarkdown{color:#c9d3e5;}[data-testid="stCaptionContainer"]{color:var(--muted)!important;}
.tl-eyebrow{color:#aaa0ff;font-size:.72rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase;margin-bottom:.7rem;}
.tl-subtitle{color:#a4b1c8;font-size:1.02rem;line-height:1.65;}
.tl-status{display:inline-flex;align-items:center;gap:7px;border:1px solid rgba(101,217,232,.25);color:#91e5ef;background:rgba(42,164,183,.09);border-radius:999px;padding:7px 12px;font-size:.72rem;font-weight:700;letter-spacing:.055em;white-space:nowrap;}
.tl-panel{background:linear-gradient(135deg,rgba(22,32,52,.96),rgba(13,21,36,.96));border:1px solid var(--border);border-radius:16px;padding:19px 22px;margin:14px 0 26px;box-shadow:0 12px 35px rgba(0,0,0,.12);}
.tl-section-kicker{color:#8f9db7;font-size:.74rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;}
.tl-verdict{display:inline-block;padding:6px 11px;border-radius:7px;font-size:.72rem;font-weight:800;letter-spacing:.08em;border:1px solid rgba(255,255,255,.12);}
.tl-supported{color:#80e3b7;background:rgba(36,171,112,.12);}.tl-contradicted{color:#ff9ba3;background:rgba(220,70,85,.12);}.tl-mixed{color:#ffd27b;background:rgba(220,160,50,.12);}.tl-inconclusive{color:#b4c0d4;background:rgba(130,145,170,.12);}
.stTextArea textarea{background:#0c1423!important;color:#f4f6ff!important;border:1px solid #2b3954!important;border-radius:13px!important;padding:17px!important;line-height:1.65;font-size:.96rem;}
.stTextArea textarea:focus{border-color:var(--violet)!important;box-shadow:0 0 0 1px var(--violet)!important;}
[data-testid="stFileUploader"]{background:#101a2b;border:1px dashed #465574;border-radius:13px;padding:12px;}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,#8b7cff,#6654dc);color:white!important;border:1px solid rgba(255,255,255,.12);border-radius:10px;min-height:46px;padding:.7rem 1.25rem;font-weight:700;transition:all .18s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{background:linear-gradient(135deg,#9b8fff,#7665ed);border-color:rgba(255,255,255,.25);transform:translateY(-1px);}
[data-testid="stMetric"]{background:linear-gradient(145deg,#172238,#101827);border:1px solid var(--border);border-radius:14px;padding:17px 18px;}
[data-testid="stMetricLabel"]{color:#9eacc4!important;}[data-testid="stMetricValue"]{color:#f5f7ff!important;}
[data-testid="stExpander"]{background:#101827;border:1px solid var(--border);border-radius:13px;overflow:hidden;}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--border)!important;border-radius:15px!important;background:rgba(16,24,39,.55);}
[data-testid="stAlert"]{border-radius:12px;}.stProgress>div>div>div{background:var(--violet);}
hr{border-color:var(--border)!important;}a{color:#b8afff!important;}
@media(max-width:900px){.block-container{padding:2rem 1.1rem 3rem!important;}}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## 🔎 TruthLens AI")
    st.caption("INVESTIGATIVE RESEARCH PROTOTYPE")
    st.markdown("---")
    st.markdown("### Project overview")
    st.write("**Gen AI for Fake News Detection**")
    st.caption("Evidence-led claim analysis using retrieval-augmented generation.")
    st.markdown("---")
    st.markdown("### System architecture")
    st.markdown("**Generation**  \nGroq API")
    st.markdown("**Embeddings**  \nSentence Transformers")
    st.markdown("**Retrieval**  \nSemantic similarity")
    st.markdown("**Verification**  \nRetrieval-Augmented Generation")
    st.markdown("---")
    st.markdown("### Capability status")
    st.markdown('<span class="tl-status">● TEXT VERIFICATION</span>', unsafe_allow_html=True)
    st.caption("Image analysis is currently a placeholder. Forensic deepfake detection is not implemented.")
    st.markdown("---")
    st.caption("TruthLens AI · Academic research prototype")

# HERO
hero_left, hero_right = st.columns([3.4, 1])
with hero_left:
    st.markdown('<div class="tl-eyebrow">AI-POWERED INFORMATION INTEGRITY</div>', unsafe_allow_html=True)
    st.title("TruthLens AI")
    st.markdown('<div class="tl-subtitle">Investigate claims. Retrieve evidence. Understand what the sources actually support.</div>', unsafe_allow_html=True)
with hero_right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align:right"><span class="tl-status">● SYSTEM READY</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="tl-panel">
<strong>Research prototype</strong>
<div style="color:#93a0b8;margin-top:5px">Assessments depend on the evidence available in the knowledge base. Results are not definitive fact-checking conclusions and should be independently reviewed.</div>
</div>
""", unsafe_allow_html=True)

# ANALYSIS WORKSPACE
st.markdown('<div class="tl-section-kicker">01 / CLAIM INTAKE</div>', unsafe_allow_html=True)
st.markdown("## Analysis workspace")
st.caption("Submit a news article or headline to extract and assess its factual claims.")
with st.container(border=True):
    st.markdown("### Article input")
    article = st.text_area("Paste a news article or headline", height=220, placeholder="Paste the news content here...")
    st.markdown("#### Supporting media")
    uploaded_image = st.file_uploader("Upload an associated news image (optional)", type=["jpg", "jpeg", "png", "webp"], help="Image processing is experimental and is not forensic deepfake detection.")
    input_col, note_col = st.columns([1, 2])
    with input_col:
        analyze_clicked = st.button("🔍  Analyze article", type="primary", use_container_width=True)
    with note_col:
        st.caption("Text analysis uses retrieved knowledge-base evidence. A minimum of 30 characters is required.")
        live_search_enabled = st.checkbox(
            "Enable live source discovery (BBC, Google Fact Check, Wikidata)",
            value=True,
            help="Live results are shown separately. News articles and Wikidata entities are not treated as verification evidence."
        )

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze_clicked:

    if len(article.strip()) < 30:
        st.warning("Please enter at least 30 characters.")
        st.stop()

    start_time = time.time()

    try:

        # Load evidence
        with st.spinner("Loading knowledge base..."):
            knowledge_base = load_knowledge_base()

        # Extract claims
        with st.spinner("Extracting factual claims..."):
            extracted = extract_claims(article)

        claims = extracted.get("claims", [])

        if not isinstance(claims, list) or not claims:
            st.warning(
                "No verifiable factual claims were extracted."
            )
            st.stop()

        st.success(f"Extracted {len(claims)} factual claims.")

        # Optional image analysis
        image_result = None

        if uploaded_image:

            try:
                with st.spinner("Processing uploaded image..."):

                    image = Image.open(
                        BytesIO(uploaded_image.getvalue())
                    )

                    image_result = analyze_image(image, article)

            except Exception as image_error:
                st.warning(
                    f"Image processing failed: {image_error}"
                )

        # Verify claims
        results = []

        progress = st.progress(0)

        for index, claim in enumerate(claims):

            try:

                with st.spinner(
                    f"Verifying claim {index + 1}/{len(claims)}..."
                ):

                    # Retrieve evidence
                    evidence = retrieve_evidence(
                        claim,
                        knowledge_base,
                        top_k=3
                    )

                    #experimental retrieval threshold.
                    #This is a provisional value and must be evaluated.
                    RETRIEVAL_THRESHOLD = 0.55

                    selected_evidence = [
                        item for item in evidence
                        if item["similarity"] >= RETRIEVAL_THRESHOLD
                    ]

                    #verify only when sufficient relevant evidence exists.
                    if selected_evidence:
                        verification = verify_claim(
                            claim,
                            selected_evidence
                        )
                    else:
                        verification = {
                            "verdict": "inconclusive",
                            "explanation": (
                                "The knowledge base did not retrieve sufficiently "
                                "relevant evidence to assess this claim. "
                                "No factual conclusion can be drawn."
                            ),
                            "confidence": None,
                            "evidence_ids": []
                        }
                    
                    # Restrict citations to retrieved evidence
                    allowed_ids = {
                        item["id"]
                        for item in selected_evidence
                    }

                    verification["evidence_ids"] = [
                        evidence_id
                        for evidence_id in verification.get(
                            "evidence_ids", []
                        )
                        if evidence_id in allowed_ids
                    ]

                    # Live-source discovery is intentionally separate from
                    # the curated RAG evidence used to determine the verdict.
                    live_sources = None
                    if live_search_enabled:
                        try:
                            live_sources = search_live_sources(claim)
                        except Exception as live_error:
                            live_sources = {
                                "fact_checks": [],
                                "news_leads": [],
                                "knowledge_graph_entities": [],
                                "provider_errors": {
                                    "live_search": f"{type(live_error).__name__}: {str(live_error)[:180]}"
                                },
                                "limitations": [
                                    "Live-source discovery failed. The local RAG result was not changed."
                                ]
                            }

                    results.append({
                        "claim": claim,
                        "verification": verification,
                        "evidence": selected_evidence,
                        "live_sources": live_sources
                    })

            except Exception as claim_error:

                # A single failed claim should not stop
                # the remaining claims from being processed.
                results.append({
                    "claim": claim,
                    "verification": {
                        "verdict": "inconclusive",
                        "explanation": (
                            "Verification could not be completed "
                            "because an error occurred: "
                            f"{str(claim_error)}"
                        ),
                        "evidence_ids": [],
                        "confidence": 0
                    },
                    "evidence": [],
                    "error": str(claim_error)
                })

            progress.progress(
                (index + 1) / len(claims)
            )

        elapsed = time.time() - start_time

        # --------------------------------------------------
        # SUMMARY CALCULATIONS
        # --------------------------------------------------

        supported = sum(
            r["verification"]["verdict"] == "supported"
            for r in results
        )

        contradicted = sum(
            r["verification"]["verdict"] == "contradicted"
            for r in results
        )

        mixed = sum(
            r["verification"]["verdict"] == "mixed"
            for r in results
        )

        inconclusive = sum(
            r["verification"]["verdict"] == "inconclusive"
            for r in results
        )

        resolved = supported + contradicted + mixed

        support_share = (
            round(
                100 * (supported + 0.5 * mixed) / resolved,
                1
            )
            if resolved else None
        )

        coverage = round(
            100 * resolved / len(results),
            1
        )

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.divider()
        st.markdown('<div class="tl-section-kicker">02 / FINDINGS</div>', unsafe_allow_html=True)
        st.markdown("## Investigation results")
        st.caption("A claim-by-claim assessment based on the evidence retrieved for this article.")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Claims extracted",
            len(claims)
        )

        col2.metric(
            "Support share",
            f"{support_share}%"
            if support_share is not None else "N/A"
        )

        col3.metric(
            "Evidence coverage",
            f"{coverage}%"
        )

        st.caption(
            "Support share is calculated from resolved claim "
            "verdicts. Evidence coverage is the percentage of "
            "claims classified as supported, contradicted or "
            "mixed. Neither metric represents the probability "
            "that the entire article is true."
        )

        # --------------------------------------------------
        # CLAIM-LEVEL VERIFICATION
        # --------------------------------------------------

        st.markdown('<div class="tl-section-kicker">03 / EVIDENCE REVIEW</div>', unsafe_allow_html=True)
        st.markdown("### Claim-by-claim verification")

        for index, result in enumerate(results):

            verification = result["verification"]
            verdict = verification.get(
                "verdict", "inconclusive"
            ).lower()

            with st.expander(
                f"{index + 1:02d}  |  {verdict.upper()}  |  {result['claim']}",
                expanded=True
            ):

                verdict_class = {
                    "supported": "tl-supported",
                    "contradicted": "tl-contradicted",
                    "mixed": "tl-mixed",
                    "inconclusive": "tl-inconclusive"
                }.get(verdict, "tl-inconclusive")
                st.markdown(
                    f'<span class="tl-verdict {verdict_class}">{verdict.upper()}</span>',
                    unsafe_allow_html=True
                )

                st.write(
                    "**Explanation:**",
                    verification.get("explanation", "")
                )

                confidence = verification.get("confidence")

                confidence_display = (
                    f"{confidence}%"
                    if confidence is not None
                    else "Not available"
                )

                st.write(
                    "**Model-reported confidence:**",
                    confidence_display
                )

                st.caption(
                    "This confidence is generated by the LLM "
                    "and is not a calibrated probability."
                )

                st.write("**Retrieved evidence:**")

                cited_ids = verification.get(
                    "evidence_ids", []
                )

                if not cited_ids:
                    st.write(
                        "No supporting evidence was identified."
                    )

                for item in result["evidence"]:

                    if item["id"] in cited_ids:

                        with st.container(border=True):
                            st.markdown(f"**{item['title']}**")
                            st.caption(item["source"])
                            st.write(item["content"])
                            st.markdown(f"[Open original source ↗]({item['url']})")

                # External discovery results are displayed separately and
                # are never passed into verify_claim() or counted as evidence.
                live = result.get("live_sources")
                if live is not None:
                    st.markdown("---")
                    st.markdown("**Live source discovery**")
                    st.caption(
                        "These are external search results, not automatically verified evidence. "
                        "They did not affect the verdict above."
                    )

                    fact_checks = live.get("fact_checks", [])
                    with st.expander(
                        f"Existing fact-check reviews · {len(fact_checks)}",
                        expanded=bool(fact_checks)
                    ):
                        if fact_checks:
                            for review in fact_checks:
                                st.markdown(f"**{review.get('title', 'Fact-check review')}**")
                                st.write(f"Publisher: {review.get('publisher', 'Unknown')}")
                                st.write(f"Published rating: {review.get('rating', 'Not supplied')}")
                                st.caption(f"Review date: {review.get('review_date') or 'Not supplied'}")
                                st.markdown(f"[Open original fact-check review ↗]({review.get('review_url', '')})")
                                st.caption(review.get("interpretation_note", ""))
                                st.markdown("---")
                        else:
                            st.caption(
                                "No indexed Google Fact Check reviews returned. "
                                "If no API key is configured, this provider is skipped."
                            )

                    news_leads = live.get("news_leads", [])
                    with st.expander(
                        f"Recent coverage to inspect · {len(news_leads)}",
                        expanded=False
                    ):
                        if news_leads:
                            for lead in news_leads:
                                st.markdown(f"**[{lead.get('title', 'Untitled article')}]({lead.get('url', '')})**")
                                st.caption(
                                    f"{lead.get('domain') or 'Unknown domain'} · "
                                    f"{lead.get('published_at') or 'Date unavailable'}"
                                    + (f" · {lead.get('language')}" if lead.get('language') else "")
                                )
                                if lead.get("summary"):
                                    st.write(lead["summary"])
                                st.caption(lead.get("interpretation_note", "News discovery lead only."))
                                st.markdown("---")
                        else:
                            st.caption("No news discovery leads returned for this claim. BBC search may have no matching articles; GDELT is optional and disabled by default.")

                    entities = live.get("knowledge_graph_entities", [])
                    with st.expander(
                        f"Related Wikidata entities · {len(entities)}",
                        expanded=False
                    ):
                        if entities:
                            for entity in entities:
                                st.markdown(f"**[{entity.get('label', 'Unnamed entity')}]({entity.get('url', '')})**")
                                if entity.get("description"):
                                    st.write(entity["description"])
                                st.caption(entity.get("interpretation_note", "Entity match only."))
                        else:
                            st.caption("No related Wikidata entities returned.")

                    provider_status = live.get("provider_status", {})
                    if provider_status:
                        st.markdown("**Provider connection status**")
                        for provider_name, status_message in provider_status.items():
                            st.caption(f"{provider_name}: {status_message}")

                    provider_errors = live.get("provider_errors", {})
                    if provider_errors:
                        st.warning("Some live providers were unavailable:")
                        for provider_name, error_message in provider_errors.items():
                            st.caption(f"{provider_name}: {error_message}")

                    st.caption(
                        "Important: GDELT coverage and Wikidata entity matches are discovery aids, "
                        "not proof that a claim is true or false. Fact-check ratings must be read "
                        "in the context of the original review, its date, and its scope."
                    )

                if result.get("error"):
                    st.warning(
                        "This claim could not be fully verified."
                    )

        # --------------------------------------------------
        # IMAGE ANALYSIS
        # --------------------------------------------------

        if image_result:

            st.divider()
            st.subheader("Image Analysis")

            st.write(
                image_result.get("description", "")
            )

            st.write(
                "**Context match:**",
                image_result.get(
                    "context_match", "inconclusive"
                )
            )

            st.write("**Visual observations:**")

            for observation in image_result.get(
                "visual_observations", []
            ):
                st.write(f"- {observation}")

            st.warning(
                "This is not a forensic deepfake detection result."
            )

            st.write("**Limitations:**")

            for limitation in image_result.get(
                "limitations", []
            ):
                st.write(f"- {limitation}")

        # --------------------------------------------------
        # FINAL SUMMARY
        # --------------------------------------------------

        st.divider()
        st.markdown('<div class="tl-section-kicker">04 / CASE SUMMARY</div>', unsafe_allow_html=True)
        st.markdown("## Verdict breakdown")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Supported", supported)
        col2.metric("Contradicted", contradicted)
        col3.metric("Mixed", mixed)
        col4.metric("Inconclusive", inconclusive)

        st.caption(
            f"Processing time: {elapsed:.2f} seconds"
        )

        # --------------------------------------------------
        # DOWNLOAD REPORT
        # --------------------------------------------------

        report = {
            "project": "TruthLens AI",
            "article": article,
            "extracted_claims": extracted,
            "claim_results": results,
            "live_source_discovery_enabled": live_search_enabled,
            "live_source_results_are_not_used_as_verification_evidence": True,
            "image_analysis": image_result,
            "support_share": support_share,
            "evidence_coverage": coverage,
            "processing_time_seconds": round(elapsed, 2)
        }

        st.download_button(
            "⬇  Export investigation report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="truthlens_analysis.json",
            mime="application/json"
        )

    except Exception as error:

        st.error(f"Analysis failed: {error}")
        st.exception(error)