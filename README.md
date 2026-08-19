# Dingo Search + AI (RAG) — Local MVP

What this is
- A local prototype search frontend renamed to "Dingo Search" that aggregates DuckDuckGo + Wikipedia (+ optional NewsAPI).
- Right-side AI assistant that follows a customizable "Guidelines" system prompt and synthesizes answers from retrieved evidence.
- Local embeddings (sentence-transformers) + FAISS for retrieval; LLM adapter supports OpenAI, Hugging Face Inference, or local TGI.

Prereqs
- Node >= 18 and npm
- Python 3.10+
- Optionally Docker if you want to run a local text-generation-inference (TGI) server for private LLM inference.

Quick start (dev)
1. Install server deps:
   cd server
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt

2. Install client deps:
   cd ../client
   npm install

3. Configure environment:
   cp ../server/.env.example ../server/.env
   Edit server/.env and add keys if you want OpenAI or NewsAPI. If you want purely local LLM, leave OPENAI_API_KEY empty and follow README section on local LLM.

4. Start backend:
   cd ../server
   source .venv/bin/activate
   uvicorn main:app --reload --port 8000

5. Start frontend:
   cd ../client
   npm run dev
   Open the Vite URL printed (default http://localhost:5173)

Notes
- Embeddings are computed locally using SentenceTransformers; first runs may download models.
- FAISS index saved under server/faiss_index.* files.
- Legal: scraping should respect robots.txt and site terms; this is a prototype only.

Privacy options
- Default: local embeddings + FAISS; LLM via OpenAI unless configured otherwise.
- To use Hugging Face Inference API: set HUGGINGFACE_API_TOKEN and MODEL_BACKEND=hf.
- To use a local TGI server: set MODEL_BACKEND=local and TGI_URL to your TGI endpoint. See server/README section for TGI example.

If you want, I can further customize the repo, add Docker Compose, or change the default LLM backend to local-only.
