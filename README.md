# Financial Legal RAG — Moroccan Law Assistant

An AI-powered chatbot for Moroccan financial and corporate law. It answers questions in French or Arabic by retrieving relevant text from a private database of legal documents — Code de Commerce, AMMC regulations, Bank Al-Maghrib circulars, company bylaws, and more — rather than relying on the model's training data alone.

---

## How it works

The system uses Retrieval-Augmented Generation (RAG). When a user asks a question, the engine first searches a local vector database for the most relevant legal excerpts, injects them into the model's context, then calls the LLM to produce a grounded answer. The model can also request a second, more targeted search before answering if it needs more context.

All communication between the engine and the LLM uses a structured JSON protocol:

```json
{
  "response_to": "user" | "system",
  "response": "...",
  "query": "..."
}
```

- `response_to: "user"` — the model is ready to answer
- `response_to: "system"` — the model wants a follow-up database search using `query`

---

## Project structure

```
.
├── src/
│   ├── rag_engine.py      # Core RAG pipeline
│   ├── server.py          # FastAPI REST server
│   └── cli.py             # Interactive terminal client
├── data/
│   ├── tmp/               # Drop documents here before ingesting
│   └── vector_stores/     # ChromaDB persisted embeddings
├── .env                   # Environment variables (see below)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

Ollama is required for local embeddings. Download it from [ollama.com](https://ollama.com), then pull the embedding model:

```bash
ollama pull qwen3-embedding:0.6b
```

> **Why qwen3-embedding:0.6b?** It handles French and Arabic text more reliably than general-purpose embedding models, which matters for Moroccan legal documents that mix both languages.

### 3. Configure environment variables

Create a `.env` file at the project root:

```env
# LLM provider: "gemini" | "openai" | "ollama"
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.1

# API keys (only the one matching your provider is needed)
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b

# Vector store
VECTOR_STORE_DIR=./data/vector_stores
K_RESULTS=4
```

### 4. Ingest your documents

Place PDF, DOCX, TXT, or CSV files in `./data/tmp`, then call the ingest endpoint:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_dir": "./data/tmp"}'
```

Or from the CLI:

```
/ingest 
```

---

## Running the server

```bash
python app/server
```

API docs are available at `http://localhost:8000/docs` once the server is running.

---

## Running the CLI

```bash
python app/cli.py
```

Available commands inside the CLI:

| Command | Description |
|---|---|
| `/ingest ` | Ingest documents from a directory |
| `/clear` | Clear conversation history |
| `/history` | Print full conversation history as JSON |
| `/quit` | Exit |

Add `--debug` to print the full JSON payload exchanged with the LLM on every turn.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send a message and get a response |
| `POST` | `/ingest` | Load documents into the vector store |
| `GET` | `/health` | Health check |
| `GET` | `/config` | Current engine configuration |

### Example chat request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "Quelles sont les obligations d'un gérant de SARL ?",
    "context": []
  }'
```

---

## Switching LLM providers

The engine supports three providers out of the box. Change `LLM_PROVIDER` in `.env`:

| Provider | `LLM_PROVIDER` | Required env var |
|---|---|---|
| Google Gemini | `gemini` | `GOOGLE_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Local (Ollama) | `ollama` | — |

> **Privacy note:** The current default uses Google Gemini, which sends every user question and retrieved legal excerpt to Google's servers. For production use with confidential client data, switch to a local Ollama model (e.g. `mistral`, `llama3`, or `qwen2.5`). This requires only a one-line change in `.env` and ensures no data ever leaves your infrastructure.

---

## Planned: automated law updates

> **Status: not yet implemented.** The `/ingest` endpoint is already in place; the automated trigger pipeline is still to be built.

Moroccan law is published through the Bulletin Officiel (Official Gazette). The planned approach is to use [n8n](https://n8n.io), an open-source workflow automation tool, to monitor the Bulletin Officiel and ministerial feeds (Ministry of Finance, AMMC, Bank Al-Maghrib) for new publications. When a relevant document is detected it will be automatically downloaded and submitted to `/ingest`, keeping the knowledge base current without manual intervention.

---

## Requirements

- Python 3.11+
- Ollama running locally (for embeddings)
- A valid API key for whichever LLM provider you choose (not needed if using Ollama for both LLM and embeddings)
