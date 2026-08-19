import os
import re
import json
import time
import hashlib
import io
import zipfile
import threading
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from bs4 import BeautifulSoup
import numpy as np
import requests

# Lazy imports for heavy libraries (load on demand)
_embedder = None
faiss = None
_embedder_lock = threading.Lock()

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "openai")  # openai | hf | local
TGI_URL = os.getenv("TGI_URL")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.json"

app = FastAPI(title="Dingo Search + AI (MVP)")

# Allow CORS from local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple domain whitelist for "trusted" badge
TRUSTED_DOMAINS = {"wikipedia.org", "nytimes.com", "bbc.co.uk", "arxiv.org", "nature.com"}

# Embedding model name (compact & fast)
EMB_MODEL_NAME = "all-MiniLM-L6-v2"

# In-memory cache for searches to reduce repeated network calls
SEARCH_CACHE: Dict[str, Dict] = {}
SEARCH_CACHE_TTL = 300  # seconds

# FAISS index placeholders
d = None
index = None
meta = []

# ---------- Lazy loaders ----------
def ensure_embeddings_loaded():
    global _embedder, faiss, d, index, meta
    # double-checked locking
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                import faiss as _faiss
                _embedder = SentenceTransformer(EMB_MODEL_NAME)
                faiss = _faiss
                d = _embedder.get_sentence_embedding_dimension()
                # initialize index
                if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
                    try:
                        index = faiss.read_index(INDEX_PATH)
                        with open(META_PATH, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        print("FAISS index loaded, entries:", len(meta))
                    except Exception as e:
                        print("Failed to load existing FAISS index:", e)
                        index = faiss.IndexFlatIP(d)
                        meta = []
                else:
                    index = faiss.IndexFlatIP(d)
                    meta = []
                print("Embeddings and FAISS initialized")


def _write_index_files(idx, meta_list):
    try:
        faiss.write_index(idx, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False, indent=2)
        print("FAISS index saved (background), entries:", len(meta_list))
    except Exception as e:
        print("Failed to save index in background:", e)


def save_index():
    global index, meta
    if index is not None:
        # write in background to avoid blocking requests
        t = threading.Thread(target=_write_index_files, args=(index, meta.copy()), daemon=True)
        t.start()


def embed_texts(texts: List[str]) -> np.ndarray:
    ensure_embeddings_loaded()
    # use the embedder to encode
    embs = _embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    return embs


def upsert_snippets(snippets: List[Dict]):
    """snippets: [{id,title,url,snippet,domain}]"""
    if not snippets:
        return
    ensure_embeddings_loaded()
    texts = [s.get("snippet") or s.get("title") or "" for s in snippets]
    embs = embed_texts(texts)
    global index, meta
    start_id = len(meta)
    for i, s in enumerate(snippets):
        s_copy = dict(s)
        s_copy["_index_id"] = start_id + i
        meta.append(s_copy)
    index.add(embs)
    # save asynchronously
    save_index()


def search_vector(query: str, top_k: int = 5):
    ensure_embeddings_loaded()
    query_emb = embed_texts([query])
    D, I = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx].copy()
        m["_score"] = float(score)
        results.append(m)
    return results

# ---------- Simple connectors with caching ----------
def cached_fetch(key: str):
    entry = SEARCH_CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > SEARCH_CACHE_TTL:
        SEARCH_CACHE.pop(key, None)
        return None
    return entry["value"]


def cached_store(key: str, value):
    SEARCH_CACHE[key] = {"ts": time.time(), "value": value}


def ddg_search(query: str, max_results=10):
    cache_key = f"ddg:{query}:{max_results}"
    cached = cached_fetch(cache_key)
    if cached is not None:
        return cached
    url = f"https://html.duckduckgo.com/html/?q={httpx.utils.escape_url(query)}"
    headers = {"User-Agent": "dingo-search-mvp/1.0 (+https://example)"}
    try:
        r = httpx.get(url, headers=headers, timeout=10.0)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for res in soup.select(".result")[:max_results]:
        a = res.find("a", {"class": "result__a"})
        title = a.get_text().strip() if a else (res.find("a").get_text().strip() if res.find("a") else "")
        href = a["href"] if a and a.has_attr("href") else None
        snippet = ""
        s = res.find(class_="result__snippet") or res.find(class_="result__summary")
        if s:
            snippet = s.get_text().strip()
        if href and title:
            try:
                domain = httpx.URL(href).host or ""
            except Exception:
                domain = ""
            items.append({"id": hashlib.sha1(href.encode()).hexdigest(), "title": title, "url": href, "snippet": snippet, "domain": domain, "source": "duckduckgo"})
    if not items:
        anchors = soup.find_all("a")[:20]
        for a in anchors:
            href = a.get("href")
            title = a.get_text().strip()
            if href and title:
                try:
                    domain = httpx.URL(href).host or ""
                except Exception:
                    domain = ""
                items.append({"id": hashlib.sha1(href.encode()).hexdigest(), "title": title, "url": href, "snippet": "", "domain": domain, "source": "duckduckgo-fallback"})
    cached_store(cache_key, items)
    return items


def wiki_search(query: str, max_results=5):
    cache_key = f"wiki:{query}:{max_results}"
    cached = cached_fetch(cache_key)
    if cached is not None:
        return cached
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": max_results}
    try:
        r = httpx.get(url, params=params, timeout=8.0)
        data = r.json()
        items = []
        for s in data.get("query", {}).get("search", []):
            title = s.get("title")
            snippet = BeautifulSoup(s.get("snippet", ""), "html.parser").get_text()
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_') }"
            domain = "wikipedia.org"
            items.append({"id": hashlib.sha1(page_url.encode()).hexdigest(), "title": title, "url": page_url, "snippet": snippet, "domain": domain, "source": "wikipedia"})
        cached_store(cache_key, items)
        return items
    except Exception:
        return []


def newsapi_search(query: str, max_results=5):
    if not NEWSAPI_KEY:
        return []
    cache_key = f"news:{query}:{max_results}"
    cached = cached_fetch(cache_key)
    if cached is not None:
        return cached
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": max_results, "apiKey": NEWSAPI_KEY}
    try:
        r = httpx.get(url, params=params, timeout=8.0)
        data = r.json()
        items = []
        for a in data.get("articles", []):
            title = a.get("title") or ""
            url = a.get("url")
            snippet = a.get("description") or ""
            try:
                domain = httpx.URL(url).host if url else ""
            except Exception:
                domain = ""
            items.append({"id": hashlib.sha1(url.encode()).hexdigest(), "title": title, "url": url, "snippet": snippet, "domain": domain, "source": "newsapi"})
        cached_store(cache_key, items)
        return items
    except Exception:
        return []

# ---------- LLM wrappers ----------

def llm_available() -> Dict:
    """Return a dict describing which LLM backends are configured and ready."""
    avail = {"openai": bool(OPENAI_KEY), "hf": bool(HF_TOKEN), "local": bool(MODEL_BACKEND == "local" and bool(TGI_URL))}
    # overall availability
    avail["any"] = avail["openai"] or avail["hf"] or avail["local"]
    avail["model"] = os.getenv("OPENAI_MODEL", OPENAI_MODEL) if avail["openai"] else (os.getenv("HF_API_URL") if avail["hf"] else (TGI_URL if avail["local"] else None))
    return avail


def call_openai_chat(messages: List[Dict], max_tokens=1024, temperature=0.2):
    if not OPENAI_KEY:
        raise RuntimeError("OpenAI key not configured")
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    j = r.json()
    return j["choices"][0]["message"]["content"]


def call_hf_text(payload_text: str):
    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token not configured")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    HF_API = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/gpt2")
    r = requests.post(HF_API, headers=headers, json={"inputs": payload_text, "parameters": {"max_new_tokens": 512, "temperature": 0.2}}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HF error {r.status_code}: {r.text}")
    j = r.json()
    if isinstance(j, list) and len(j) > 0:
        return j[0].get("generated_text", "")
    return ""


def call_local_tgi(prompt: str):
    if not TGI_URL:
        raise RuntimeError("TGI_URL not configured for local backend")
    r = requests.post(TGI_URL.rstrip("/") + "/generate", json={"prompt": prompt, "max_new_tokens": 1024}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"TGI error {r.status_code}: {r.text}")
    return r.json().get("generated_text", "")


def run_llm(with_messages: List[Dict], concatenated_evidence: str, system_guidelines: str):
    # Build a clear instruction that forces citing evidence blocks and respects guidelines
    instruction = (
        "You are Dingo.ai, a helpful assistant that summarizes high-quality sources. Using only the evidence blocks below, answer the user's last question concisely and include numeric citations like [E1], [E2]. "
        "If evidence doesn't support a claim, say 'no reliable source found'. Do not invent facts. Do not produce image-generation requests."
        "\n\nEvidence:\n" + concatenated_evidence + "\n\nNow answer:"
    )
    messages = []
    if system_guidelines:
        messages.append({"role": "system", "content": system_guidelines})
    for m in with_messages:
        messages.append(m)
    messages.append({"role": "system", "content": instruction})
    avail = llm_available()
    if not avail.get("any"):
        raise RuntimeError("No LLM backend configured. Set OPENAI_API_KEY or HUGGINGFACE_API_TOKEN or configure local TGI.")
    if MODEL_BACKEND == "openai" and OPENAI_KEY:
        return call_openai_chat(messages)
    elif MODEL_BACKEND == "hf" and HF_TOKEN:
        prompt = "\n".join([m.get("content", "") for m in messages])
        return call_hf_text(prompt)
    elif MODEL_BACKEND == "local":
        prompt = "\n".join([m.get("content", "") for m in messages])
        return call_local_tgi(prompt)
    else:
        # fallback to any available
        if OPENAI_KEY:
            return call_openai_chat(messages)
        if HF_TOKEN:
            prompt = "\n".join([m.get("content", "") for m in messages])
            return call_hf_text(prompt)
        if MODEL_BACKEND == "local" and TGI_URL:
            prompt = "\n".join([m.get("content", "") for m in messages])
            return call_local_tgi(prompt)
        raise RuntimeError("No supported LLM backend could be used.")

# ---------- Utility ----------
def trust_score_for_domain(domain: str) -> float:
    if not domain:
        return 0.1
    for d in TRUSTED_DOMAINS:
        if d in domain:
            return 0.95
    # deprioritize code hosting domains to avoid code-only results
    if domain and ("github.com" in domain or "gitlab.com" in domain or "bitbucket.org" in domain):
        return 0.2
    return 0.5

# ---------- API models ----------
class SearchResponseItem(BaseModel):
    id: str
    title: str
    url: str
    snippet: Optional[str] = ""
    domain: Optional[str] = ""
    source: Optional[str] = ""
    trust: Optional[float] = 0.0

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResponseItem]

class ChatRequest(BaseModel):
    messages: List[Dict]  # basic {role,content}
    systemGuidelines: Optional[str] = ""
    selectedEvidenceIds: Optional[List[str]] = []
    screen_text: Optional[str] = ""  # text captured from the user's visible app screen

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict]

# ---------- Endpoints ----------
@app.get("/api/health")
def api_health():
    """Return basic health and LLM/embeddings readiness information."""
    emb_ready = False
    try:
        if _embedder is not None:
            emb_ready = True
        else:
            emb_ready = False
    except Exception:
        emb_ready = False
    llm = llm_available()
    return {"embeddings_loaded": emb_ready, "llm": llm}


@app.get("/api/search", response_model=SearchResponse)
async def api_search(q: str = Query(..., min_length=1)):
    # Try cache first across connectors
    cache_key = f"combined:{q}"
    cached = cached_fetch(cache_key)
    if cached is not None:
        return {"query": q, "results": cached}

    # Knowledge-first profile: prioritize wiki and news for general queries
    ddg = ddg_search(q, max_results=6)
    wiki = wiki_search(q, max_results=3)
    news = newsapi_search(q, max_results=2)
    combined = ddg + wiki + news
    seen = set()
    unique = []
    for it in combined:
        if not it.get("url"):
            continue
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        it["trust"] = trust_score_for_domain(it.get("domain", ""))
        unique.append(it)
    upsert_snippets(unique)
    # reorder: prefer trusted domains then highest trust then recency heuristic (not implemented fully)
    unique.sort(key=lambda x: (-x.get("trust", 0.0)))
    resp_items = []
    for u in unique[:20]:
        resp_items.append(SearchResponseItem(**u))
    # cache combined results briefly
    cached_store(cache_key, resp_items)
    return {"query": q, "results": resp_items}


@app.get("/api/summarize")
def api_summarize(q: str = Query(..., min_length=1)):
    """Produce a Dingo.ai style summary of top retrieved evidence for a query."""
    # perform same retrieval as search but focused and deduped
    ddg = ddg_search(q, max_results=6)
    wiki = wiki_search(q, max_results=3)
    news = newsapi_search(q, max_results=2)
    combined = ddg + wiki + news
    seen = set()
    unique = []
    for it in combined:
        if not it.get("url"):
            continue
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        it["trust"] = trust_score_for_domain(it.get("domain", ""))
        unique.append(it)
    # upsert snippets for vector retrieval
    upsert_snippets(unique)
    # retrieve top evidence from vector db for the query
    evidence = []
    try:
        retrieved = search_vector(q, top_k=6)
    except Exception:
        retrieved = []
    for r in retrieved:
        evidence.append({"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet"), "trust": r.get("trust", 0.5)})

    concatenated = ""
    for i, e in enumerate(evidence, start=1):
        concatenated += f"[E{i}] {e.get('title','')}\nURL: {e.get('url')}\nSnippet: {e.get('snippet','')}\n\n"

    # ask LLM to summarize
    user_messages = [{"role": "user", "content": q}]
    system_guidelines = "You are Dingo.ai, a concise summarizer. Cite evidence like [E1]. Do not invent facts."
    try:
        summary = run_llm(user_messages, concatenated, system_guidelines)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {"query": q, "summary": summary, "evidence": evidence}


@app.get("/api/retrieve")
def api_retrieve(url: str):
    try:
        r = httpx.get(url, timeout=12)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch URL")
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
        text = "\n\n".join(paragraphs[:200])
        return {"url": url, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    selected = req.selectedEvidenceIds or []
    evidence_blocks = []
    # If the client provided screen_text, include it as E0 (highest priority)
    if req.screen_text:
        evidence_blocks.append({"id": "_screen_", "url": "", "snippet": req.screen_text, "title": "Screen capture (user)", "trust": 0.6})

    if selected:
        id_map = {m["id"]: m for m in meta}
        for sid in selected:
            if sid in id_map:
                m = id_map[sid]
                evidence_blocks.append({"id": m["id"], "url": m["url"], "snippet": m.get("snippet", ""), "title": m.get("title", ""), "trust": m.get("trust", 0.5)})
    else:
        last_user = None
        for m in reversed(req.messages):
            if m.get("role") == "user":
                last_user = m.get("content")
                break
        if not last_user:
            raise HTTPException(status_code=400, detail="No user message provided")
        retrieved = []
        try:
            retrieved = search_vector(last_user, top_k=6)
        except Exception:
            retrieved = []
        for r in retrieved:
            evidence_blocks.append({"id": r.get("id"), "url": r.get("url"), "snippet": r.get("snippet"), "title": r.get("title"), "trust": r.get("trust", 0.5), "score": r.get("_score", 0.0)})

    concatenated = ""
    for i, e in enumerate(evidence_blocks, start=1):
        concatenated += f"[E{i}] {e.get('title','')}\nURL: {e.get('url')}\nSnippet: {e.get('snippet','')}\n\n"

    try:
        ans = run_llm(req.messages, concatenated, req.systemGuidelines or "")
    except Exception as exc:
        fallback = "\n\n".join([f"{i+1}. {e.get('title','')} - {e.get('snippet','')} ({e.get('url')})" for i, e in enumerate(evidence_blocks)])
        raise HTTPException(status_code=503, detail=f"LLM error: {str(exc)}. Fallback evidence:\n{fallback}")

    citations = []
    for i, e in enumerate(evidence_blocks, start=1):
        citations.append({"label": f"[E{i}]", "url": e.get("url"), "title": e.get("title"), "snippet": e.get("snippet"), "trust": e.get("trust")})
    return {"answer": ans, "citations": citations}


@app.post("/api/build_site")
def api_build_site(payload: Dict = Body(...)):
    """Build a static SEO-optimized site for a given topic and return a ZIP.
    Expected payload: { "topic": "...", "tone": "...", "length": "short|medium|long" }
    """
    topic = payload.get("topic")
    if not topic:
        raise HTTPException(status_code=400, detail="Missing topic")
    tone = payload.get("tone", "informative")
    length = payload.get("length", "medium")

    # retrieve knowledge-first evidence
    ddg = ddg_search(topic, max_results=6)
    wiki = wiki_search(topic, max_results=4)
    news = newsapi_search(topic, max_results=3)
    combined = ddg + wiki + news
    seen = set()
    unique = []
    for it in combined:
        if not it.get("url"):
            continue
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        it["trust"] = trust_score_for_domain(it.get("domain", ""))
        unique.append(it)
    upsert_snippets(unique)

    # gather evidence for LLM
    retrieved = []
    try:
        retrieved = search_vector(topic, top_k=8)
    except Exception:
        retrieved = []
    evidence_text = ""
    for i, r in enumerate(retrieved, start=1):
        evidence_text += f"[E{i}] {r.get('title','')}\nURL: {r.get('url')}\nSnippet: {r.get('snippet','')}\n\n"

    # prompt LLM to output a JSON structure of files for the static site
    system_guidelines = ("You are Dingo.ai, an assistant that generates high-quality, SEO-optimized static sites. "
                         "Produce a JSON object with a top-level 'files' array where each entry is {\"path\": ..., \"content\": ...}. "
                         "Only return valid JSON. Include index.html, styles.css, and optionally other assets. Keep content original and cite evidence where appropriate.")
    user_prompt = (
        f"Generate a static site for the topic: {topic}. Tone: {tone}. Length: {length}. "
        f"Use the evidence below to ground facts. Evidence:\n{evidence_text}\n\nReturn only a JSON object with 'files'."
    )

    try:
        llm_out = run_llm([{"role": "user", "content": user_prompt}], evidence_text, system_guidelines)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM error building site: {exc}")

    files = None
    try:
        # Attempt to parse JSON from LLM output
        files_obj = json.loads(llm_out)
        files = files_obj.get("files")
    except Exception:
        # fallback: create a single index.html with the LLM output as content
        index_html = f"<html><head><meta charset=\"utf-8\"><title>{topic}</title></head><body><pre>{html_escape(llm_out)}</pre></body></html>"
        files = [{"path": "index.html", "content": index_html}]

    # build ZIP in-memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            path = f.get("path")
            content = f.get("content")
            if not path or content is None:
                continue
            zf.writestr(path, content)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=site_{hashlib.sha1(topic.encode()).hexdigest()[:8]}.zip"})


# small helper for fallback
def html_escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
