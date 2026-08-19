# Dingo Search + AI (RAG) — Local MVP (updated run instructions)

What this is
- A local prototype search frontend renamed to "Dingo Search" that aggregates DuckDuckGo + Wikipedia (+ optional NewsAPI).
- Right-side AI assistant that follows a customizable "Guidelines" system prompt and synthesizes answers from retrieved evidence.
- Local embeddings (sentence-transformers) + FAISS for retrieval; LLM adapter supports OpenAI, Hugging Face Inference, or local TGI.

Prereqs
- Node >= 18 and npm
- Python 3.10+
- Optionally Docker if you want to run a local text-generation-inference (TGI) server for private LLM inference.

Quick start (dev)
1. From the project root, install both client JS deps and server Python deps (this will run automatically before the app starts):
   npm start

   The `prestart` script runs `npm run install:all` which installs client npm packages and server Python requirements.

2. If you prefer manual install:
   - Install server Python deps:
     cd server
     python -m venv .venv
     source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
     python -m pip install -r requirements.txt

   - Install client deps:
     cd ../client
     npm install

3. Start backend (if running manually):
   cd server
   python -m uvicorn main:app --reload --port 8000

4. Start frontend (if running manually):
   cd client
   npm run dev

Notes
- If you see errors about `vite` or `uvicorn` not found, make sure you've completed the install step. The automated `npm start` runs the install step first.
- `npm start` will run both the frontend (Vite dev) and the backend (uvicorn) concurrently.

Privacy & options
- Embeddings are computed locally using SentenceTransformers by default; model downloads happen on first use.
- To avoid external LLMs, set MODEL_BACKEND=local and run a local TGI instance; see server/.env.example.

If you want me to change the startup behavior (e.g., separate containers, Docker Compose), I can add Docker files next.
