import os
import re
import json
import time
import hashlib
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "openai")  # openai | hf | local
TGI_URL = os.getenv("TGI_URL")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.json"

app = FastAPI(title="DuckLike Search + AI (MVP)")

# Simple domain whitelist for "trusted" badge
TRUSTED_DOMAINS = {"wikipedia.org", "nytimes.com", "bbc.co.uk", "arxiv.org", "nature.com"}

# Load sentence-transformers model (local embeddings)
EMB_MODEL_NAME = "all-MiniLM-L6-v2"  # compact and fast
embedder = SentenceTransformer(EMB_MODEL_NAME)

# FAISS index: we'll use cosine similarity via normalized vectors
d = embedder.get_sentence_embedding_dimension()
index = None
meta = []

def load_index():
    global index, meta
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print("FAISS index loaded, entries:", len(meta))
            return
        except Exception as e:
            print("Failed to load index:", e)
    # create new
    index = faiss.IndexFlatIP(d)  # inner product; we'll normalize vectors
    meta = []
    print("Created new FAISS index")

def save_index():
    if index is not None:
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("FAISS index saved, entries:", len(meta))

def embed_texts(texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # normalize
    faiss.normalize_L2(embs)
    return embs

def upsert_snippets(snippets: List[Dict]):
    """snippets: [{id,title,url,snippet,domain}]"""
    if not snippets:
        return
    texts = [s.get("snippet") or s.get("title") or "" for s in snippets]
    embs = embed_texts(texts)
    global index, meta
    start_id = len(meta)
    # append metadata
    for i, s in enumerate(snippets):
        s_copy = dict(s)
        s_copy["_index_id"] = start_id + i
        meta.append(s_copy)
    # add to index
    index.add(embs)
    save_index()

def search_vector(query: str, top_k: int = 5):
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

# ---------- Simple connectors ----------
def ddg_search(query: str, max_results=10):
    url = f"https://html.duckduckgo.com/html/?q={httpx.utils.escape_url(query)}"
    headers = {"User-Agent": "ducklike-mvp/1.0 (+https://example)"}
    r = httpx.get(url, headers=headers, timeout=15.0)
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
            domain = httpx.URL(href).host or ""
            items.append({"id": hashlib.sha1(href.encode()).hexdigest(), "title": title, "url": href, "snippet": snippet, "domain": domain, "source": "duckduckgo"})
    # fallback: basic anchor scan
    if not items:
        anchors = soup.find_all("a")[:20]
        for a in anchors:
            href = a.get("href")
            title = a.get_text().strip()
            if href and title:
                domain = httpx.URL(href).host or ""
                items.append({"id": hashlib.sha1(href.encode()).hexdigest(), "title": title, "url": href, "snippet": "", "domain": domain, "source": "duckduckgo-fallback"})
    return items

def wiki_search(query: str, max_results=5):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": max_results}
    try:
        r = httpx.get(url, params=params, timeout=10.0)
        data = r.json()
        items = []
        for s in data.get("query", {}).get("search", []):
            title = s.get("title")
            snippet = BeautifulSoup(s.get("snippet", ""), "html.parser").get_text()
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_') }"
            domain = "wikipedia.org"
            items.append({"id": hashlib.sha1(page_url.encode()).hexdigest(), "title": title, "url": page_url, "snippet": snippet, "domain": domain, "source": "wikipedia"})
        return items
    except Exception as e:
        print("Wiki search error:", e)
        return []

def newsapi_search(query: str, max_results=5):
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": max_results, "apiKey": NEWSAPI_KEY}
    try:
        r = httpx.get(url, params=params, timeout=10.0)
        data = r.json()
        items = []
        for a in data.get("articles", []):
            title = a.get("title") or ""
            url = a.get("url")
            snippet = a.get("description") or ""
            domain = httpx.URL(url).host if url else ""
            items.append({"id": hashlib.sha1(url.encode()).hexdigest(), "title": title, "url": url, "snippet": snippet, "domain": domain, "source": "newsapi"})
        return items
    except Exception as e:
        print("NewsAPI error:", e)
        return []

# ---------- LLM wrappers ----------
def call_openai_chat(messages: List[Dict], max_tokens=512, temperature=0.2):
    if not OPENAI_KEY:
        raise RuntimeError("OpenAI key not configured")
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    j = r.json()
    return j["choices"][0]["message"]["content"]

def call_hf_text(payload_text: str):
    if not HF_TOKEN:
        raise RuntimeError("Hugging Face token not configured")
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    # Using text generation endpoint (this example uses the "text-generation" API)
    HF_API = "https://api-inference.huggingface.co/models/gpt2"  # replace with desired model or env var
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
    r = requests.post(TGI_URL.rstrip("/") + "/generate", json={"prompt": prompt, "max_new_tokens": 512}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"TGI error {r.status_code}: {r.text}")
    return r.json().get("generated_text", "")

def run_llm(with_messages: List[Dict], concatenated_evidence: str, system_guidelines: str):
    # Build a simple instruction that forces citing evidence blocks
    instruction = "Using only the evidence blocks below, answer the user's last question concisely and include numeric citations like [E1], [E2]. If evidence doesn't support a claim, say 'no reliable source found'.\n\nEvidence:\n" + concatenated_evidence + "\n\nNow answer:"
    # Build messages to LLM
    messages = []
    if system_guidelines:
        messages.append({"role":"system","content": system_guidelines})
    # append user's conversation (we expect with_messages to include user's last message)
    for m in with_messages:
        messages.append(m)
    # append instruction as system/assistant message
    messages.append({"role":"system","content": instruction})
    # choose backend
    if MODEL_BACKEND == "openai" and OPENAI_KEY:
        return call_openai_chat(messages)
    elif MODEL_BACKEND == "hf" and HF_TOKEN:
        # Flatten prompt
        prompt = "\n".join([m.get("content","") for m in messages])
        return call_hf_text(prompt)
    elif MODEL_BACKEND == "local":
        prompt = "\n".join([m.get("content","") for m in messages])
        return call_local_tgi(prompt)
    else:
        raise RuntimeError("No LLM backend configured. Set OPENAI_API_KEY or HUGGINGFACE_API_TOKEN or configure local TGI.")

# ---------- Utility ----------
def trust_score_for_domain(domain: str) -> float:
    if not domain:
        return 0.1
    for d in TRUSTED_DOMAINS:
        if d in domain:
            return 0.95
    # simple heuristic: longer domains may be more established
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

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict]

# ---------- Endpoints ----------
@app.on_event("startup")
def startup_event():
    load_index()

@app.get("/api/search", response_model=SearchResponse)
async def api_search(q: str = Query(..., min_length=1)):
    # call connectors
    ddg = ddg_search(q, max_results=8)
    wiki = wiki_search(q, max_results=4)
    news = newsapi_search(q, max_results=3)
    combined = ddg + wiki + news
    # simple dedupe by url hash
    seen = set()
    unique = []
    for it in combined:
        if not it.get("url"):
            continue
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        it["trust"] = trust_score_for_domain(it.get("domain",""))
        unique.append(it)
    # upsert snippets into vector DB for later retrieval
    upsert_snippets(unique)
    # sort by trust + source (simple)
    unique.sort(key=lambda x: (-x.get("trust",0.0)))
    # map to response items (limit)
    resp_items = []
    for u in unique[:20]:
        resp_items.append(SearchResponseItem(**u))
    return {"query": q, "results": resp_items}

@app.get("/api/retrieve")
def api_retrieve(url: str):
    # fetch and return parsed text (basic)
    try:
        r = httpx.get(url, timeout=15)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch URL")
        soup = BeautifulSoup(r.text, "html.parser")
        # attempt to extract readable text
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
        text = "\n\n".join(paragraphs[:200])
        return {"url": url, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    # gather evidence: either from selectedEvidenceIds or perform vector search using last user message
    selected = req.selectedEvidenceIds or []
    evidence_blocks = []
    if selected:
        # find metadata by id
        id_map = {m["id"]: m for m in meta}
        for sid in selected:
            if sid in id_map:
                m = id_map[sid]
                evidence_blocks.append({"id": m["id"], "url": m["url"], "snippet": m.get("snippet",""), "title": m.get("title",""), "trust": m.get("trust", 0.5)})
    else:
        # no selected ids: use last user message to query vector store
        last_user = None
        for m in reversed(req.messages):
            if m.get("role") == "user":
                last_user = m.get("content")
                break
        if not last_user:
            raise HTTPException(status_code=400, detail="No user message provided")
        retrieved = search_vector(last_user, top_k=6)
        for r in retrieved:
            evidence_blocks.append({"id": r.get("id"), "url": r.get("url"), "snippet": r.get("snippet"), "title": r.get("title"), "trust": r.get("trust", 0.5), "score": r.get("_score", 0.0)})

    # build concatenated evidence text
    concatenated = ""
    for i, e in enumerate(evidence_blocks, start=1):
        concatenated += f"[E{i}] {e.get('title','')}\nURL: {e.get('url')}\nSnippet: {e.get('snippet','')}\n\n"

    # call LLM
    try:
        ans = run_llm(req.messages, concatenated, req.systemGuidelines or "")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # return structured result
    citations = []
    for i, e in enumerate(evidence_blocks, start=1):
        citations.append({"label": f"[E{i}]", "url": e.get("url"), "title": e.get("title"), "snippet": e.get("snippet"), "trust": e.get("trust")})
    return {"answer": ans, "citations": citations}
