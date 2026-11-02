# TruthLens AI

### Generative AI for Fake News Detection: Combating AI-Generated Misinformation

TruthLens AI is an academic research prototype that helps users
investigate potentially misleading news articles. It extracts factual
claims from submitted text, searches a curated evidence collection, and
uses a language model to produce evidence-grounded assessments.

> **Important:** TruthLens is a decision-support and research prototype,
> not an automated authority on truth. Its conclusions depend on the
> quality, coverage, and relevance of the available evidence. An
> **Inconclusive** result means the system could not establish or
> directly contradict a claim using the evidence it retrieved; it does
> not mean the claim is false.

------------------------------------------------------------------------

## 1. Project overview

Online misinformation can spread quickly, and manually checking every
factual statement in an article takes time. TruthLens AI explores a
pipeline that combines:

-   Natural-language processing to identify factual claims.
-   Retrieval-Augmented Generation (RAG) to find relevant passages in a
    curated knowledge base.
-   A large language model (LLM) to assess claims against the retrieved
    material.
-   External information discovery to help locate potentially relevant
    reporting and entity information.
-   Source references and structured output to make the analysis easier
    to inspect.

The system analyses text supplied by the user. It is designed to support
human investigation, not replace independent fact-checking.

## 2. What the application does

A typical analysis follows these steps:

1.  **Article input** --- The user pastes article text into the
    Streamlit interface.
2.  **Claim extraction** --- The language model identifies individual
    factual claims in the text.
3.  **Local evidence retrieval** --- Each claim is converted into an
    embedding using `sentence-transformers/all-MiniLM-L6-v2`. The RAG
    service compares it with embeddings of passages in the local
    evidence collection and retrieves the most similar passages.
4.  **Evidence threshold** --- The current application uses a
    provisional similarity threshold of `0.55`. If no retrieved passage
    meets the threshold, the claim is not sent to the LLM for a verdict;
    it is returned as inconclusive.
5.  **Claim assessment** --- When suitable passages are available, the
    LLM receives the claim and selected evidence and is instructed to
    assess whether the evidence supports, directly contradicts, or does
    not establish the claim.
6.  **Result display** --- The interface presents claim-level
    assessments and the associated evidence.
7.  **Export** --- The analysis can be exported as JSON.

The threshold is an experimental setting, not a scientifically validated
universal cutoff.

## 3. Verdict interpretation

  -----------------------------------------------------------------------
  Result                              Meaning
  ----------------------------------- -----------------------------------
  **Supported**                       The retrieved evidence supports the
                                      claim.

  **Contradicted**                    The retrieved evidence directly
                                      conflicts with the claim.

  **Inconclusive**                    The available retrieved evidence
                                      does not establish or directly
                                      contradict the claim.
  -----------------------------------------------------------------------

The system follows a conservative evidence rule: lack of supporting
evidence is not, by itself, proof that a claim is false. Evidence about
a general topic should not be treated as proof against a specific claim
unless it directly conflicts with that claim.

Any confidence value produced by the language model should be treated as
a model estimate, not as a calibrated probability that the verdict is
correct.

## 4. Architecture

``` text
                  USER
                   |
                   v
          Streamlit web interface
                   |
                   v
             Article text
                   |
                   v
             Claim extraction
                   |
                   v
          Individual factual claims
                   |
                   v
          Local RAG evidence retrieval
                   |
          Similarity threshold: 0.55
                   |
          +--------+---------+
          |                  |
     Evidence found     No suitable evidence
          |                  |
          v                  v
   LLM claim assessment   Inconclusive
          |
          v
   Evidence-grounded verdict
          |
          v
   Results, source references
          |
          v
        JSON export
```

### External discovery path

External discovery is separate from the permanent local knowledge base:

``` text
Claim
  |
  v
Live evidence service
  |
  +--> BBC News search (news discovery leads)
  |
  +--> Google Fact Check Tools API (optional; requires API key)
  |
  +--> Wikidata (related entity information)
  |
  +--> GDELT (optional; disabled by default)
```

These providers have different roles:

-   **BBC News:** Finds candidate news articles. Search results are
    discovery leads, not verified evidence.
-   **Google Fact Check Tools:** Can return existing fact-check reviews
    when configured with an API key.
-   **Wikidata:** Returns related entities and structured descriptions.
    An entity match is not proof of a claim.
-   **GDELT:** Optional news discovery provider. It is disabled by
    default and may be unavailable from some networks.

Live discovery results are kept separate from the curated local evidence
collection. They are not automatically treated as proof or permanently
added to the knowledge base.

## 5. Technology stack

  ------------------------------------------------------------------------------------------
  Component               Technology                                 Purpose
  ----------------------- ------------------------------------------ -----------------------
  User interface          Streamlit                                  Interactive web
                                                                     application

  Main language           Python                                     Application and service
                                                                     logic

  LLM provider            Groq API                                   Claim extraction and
                                                                     evidence-grounded
                                                                     assessment

  Configured LLM          `openai/gpt-oss-120b` (subject to `.env`   Language-model tasks
                          configuration)                             

  Embeddings              Sentence Transformers                      Convert claims and
                                                                     evidence passages into
                                                                     vectors

  Embedding model         `sentence-transformers/all-MiniLM-L6-v2`   Local 384-dimensional
                                                                     sentence embeddings

  Similarity search       Cosine similarity                          Rank evidence passages
                                                                     against a claim

  HTTP requests           Requests                                   Connect to external
                                                                     services

  HTML parsing            Beautiful Soup                             Parse BBC search
                                                                     results

  Local evidence store    JSON files                                 Store curated evidence
                                                                     and source metadata

  Configuration           python-dotenv / `.env`                     Load secrets and
                                                                     environment settings

  Tests                   Python `unittest`                          Test live evidence
                                                                     service behaviour
  ------------------------------------------------------------------------------------------

The embedding model runs locally. The configured Groq LLM is accessed
through the API, so claim extraction and assessment require network
access and a valid API key.

## 6. Knowledge base

The local evidence collection is stored at:

``` text
data/knowledge_base/evidence.json
```

At the time of this README, the collection contains **262 evidence
passages**:

-   260 passages ingested from eight World Health Organization (WHO)
    source pages.
-   2 original NASA-related evidence records.

The WHO source topics include:

-   Tobacco
-   Vaccines
-   Climate
-   Malaria
-   Tuberculosis
-   Diabetes
-   Obesity
-   Air pollution

The 262 records are passages, not 262 independent sources. Multiple
passages may originate from the same source page.

### Ingestion and validation

The project includes scripts for collecting, validating, and merging
evidence:

``` text
scripts/
├── ingestion/
│   └── ingest_sources.py
└── validation/
    ├── validate_evidence.py
    └── merge_evidence.py
```

The validation workflow checks the structure and quality of ingested
records before they are merged into the main evidence file. Backups of
the evidence collection are stored under:

``` text
data/knowledge_base/backups/
```

Do not assume that a source is authoritative merely because it was
successfully ingested. Source quality and relevance still need to be
evaluated.

## 7. Project structure

``` text
genai-fake-news-detection/
│
├── app.py
├── .env
├── .gitignore
├── README.md
│
├── backend/
│   ├── __init__.py
│   ├── core/
│   │   └── __init__.py
│   └── services/
│       ├── __init__.py
│       ├── gemini_service.py
│       ├── rag_service.py
│       └── live_evidence_service.py
│
├── data/
│   ├── knowledge_base/
│   │   ├── evidence.json
│   │   └── backups/
│   ├── processed/
│   │   ├── who_evidence.json
│   │   └── validation_report.json
│   └── source_registry.json
│
├── scripts/
│   ├── ingestion/
│   │   └── ingest_sources.py
│   ├── validation/
│   │   ├── validate_evidence.py
│   │   └── merge_evidence.py
│   └── tests/
│       └── test_live_evidence_service.py
│
└── .venv/
```

**Note:** `gemini_service.py` is a legacy filename retained for
compatibility with existing imports. The current LLM implementation in
that service uses Groq.

## 8. Setup and installation

### Requirements

-   Python 3.10 or later is recommended.
-   Git
-   Internet connection for the Groq API and live discovery providers.
-   A Groq API key.
-   Enough disk space to download the sentence-transformer model.

### Clone the repository

``` bash
git clone <REPOSITORY_URL>
cd genai-fake-news-detection
```

Replace `<REPOSITORY_URL>` with the actual repository URL shared by the
project team.

### Create and activate a virtual environment

Linux / WSL:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

``` powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

If the repository contains a `requirements.txt` file:

``` bash
pip install -r requirements.txt
```

If it does not, install the packages used by the current application:

``` bash
pip install streamlit groq sentence-transformers torch numpy requests beautifulsoup4 python-dotenv
```

The first run may download the embedding model. Allow time for that
download.

### Configure environment variables

Create a `.env` file in the project root:

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Optional providers:

``` env
GOOGLE_FACTCHECK_API_KEY=your_google_factcheck_api_key
ENABLE_GDELT=false
```

Use the exact Google Fact Check environment-variable name expected by
the current `live_evidence_service.py` implementation if it differs from
the example above.

**Security:** Never commit `.env` or share API keys in screenshots,
reports, or public repositories. Keep `.env` in `.gitignore`. Each team
member should configure their own key locally.

## 9. Run the application

From the project root, with the virtual environment activated:

``` bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

``` text
http://localhost:8501
```

Open that address in a browser.

## 10. Run tests

Run the live evidence service tests:

``` bash
python -m unittest discover -s scripts/tests -v
```

Check Python syntax:

``` bash
python -m py_compile app.py backend/services/*.py
```

The live connector tests use mocked responses to check expected
behaviour. Passing these tests does not guarantee that an external
provider is reachable or that its live search results are relevant.

## 11. Current evaluation and observations

The local retrieval system has been tested with a small set of manually
prepared in-domain and out-of-domain claims.

Examples of observed behaviour:

-   WHO-topic claims such as malaria, tuberculosis, tobacco, vaccines,
    diabetes, obesity, and air pollution retrieve relevant passages in
    the current test set.
-   Unrelated claims, such as cricket results or questions outside the
    knowledge base, tend to receive low similarity scores.
-   The application uses a provisional similarity threshold of `0.55` to
    avoid sending unsupported claims to the LLM for a confident-sounding
    answer.

These are preliminary experiments, not a comprehensive benchmark. They
do not establish general accuracy across topics, languages, publishers,
or types of misinformation.

## 12. Known limitations

1.  **Limited knowledge-base coverage:** The curated collection is
    mainly WHO health material plus a small number of NASA records.
    Claims outside those topics may be inconclusive.
2.  **Similarity threshold is provisional:** `0.55` is an experimental
    cutoff and has not been universally validated.
3.  **LLM limitations:** Language models can misinterpret evidence or
    generate errors. Their confidence estimates are not calibrated
    probabilities.
4.  **Live news discovery is not fact verification:** BBC results may be
    loosely related to a search query. They must be opened and assessed
    by a human before being used as evidence.
5.  **Provider availability:** External services can time out, change
    their page structure, impose quotas, or require API keys.
6.  **Image analysis is not implemented:** The current image-analysis
    function is a stub. Uploading an image does not provide genuine
    image forensics, deepfake detection, or reliable visual
    verification.
7.  **No universal fake-news detector:** The system does not establish
    whether an entire article is fake. It assesses extracted claims
    against the evidence it can retrieve.

## 13. Responsible use

-   Read the cited source material before making a final judgement.
-   Treat **Inconclusive** as "not established with the available
    evidence," not "false."
-   Do not present model-generated confidence as a statistical
    probability.
-   Do not treat a news-search result, entity description, or search
    snippet as proof.
-   Verify important claims using multiple credible, relevant sources.
-   Do not use the prototype as the sole basis for high-impact
    decisions.

## 14. Suggested demonstration flow

For a project demonstration:

1.  Start the Streamlit application.
2.  Paste an article containing several factual statements.
3.  Run the analysis.
4.  Show the claims extracted from the article.
5.  Open a claim result and explain the retrieved evidence.
6.  Demonstrate an **Inconclusive** result for a claim outside the local
    knowledge-base coverage.
7.  Show the live discovery section, explaining the difference between
    news leads, fact-check reviews, and entity information.
8.  Export the analysis as JSON.
9.  Explain the current limitations, especially knowledge-base coverage
    and the image-analysis stub.

## 15. Team contribution areas

The project can be understood as the following work areas:

-   **Frontend and application:** Streamlit interface, user input,
    result display, and export.
-   **NLP and LLM:** Claim extraction and evidence-grounded claim
    assessment.
-   **RAG and embeddings:** Evidence ingestion, vectorisation,
    similarity retrieval, and thresholding.
-   **Data and source management:** Source registry, evidence
    validation, provenance, and knowledge-base maintenance.
-   **External integrations and testing:** BBC, Google Fact Check,
    Wikidata, optional GDELT, and connector tests.

Team members should document their actual contributions and avoid
claiming work they did not perform.

------------------------------------------------------------------------

## Project status

TruthLens AI is an evolving academic prototype. Its current
implementation demonstrates a complete text-analysis workflow using
claim extraction, local evidence retrieval, LLM-assisted assessment,
external information discovery, and JSON export. Its reliability is
constrained by the scope of the evidence collection, retrieval
threshold, external provider behaviour, and language-model limitations.

**The goal is evidence-assisted investigation --- not blindly trusting
an AI verdict.**
