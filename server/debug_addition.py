# debug endpoint: expose basic runtime info
@app.get('/api/debug')
def api_debug():
    try:
        return {
            "llm": llm_available(),
            "embeddings_loaded": _embedder is not None,
            "faiss_entries": len(meta) if isinstance(meta, list) else 0,
            "env": {"OPENAI_KEY_SET": bool(OPENAI_KEY), "MODEL_BACKEND": MODEL_BACKEND, "TGI_URL_SET": bool(TGI_URL)}
        }
    except Exception as e:
        return {"error": str(e)}
