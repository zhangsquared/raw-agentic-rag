# Raw Agentic RAG

User LangChain to buid an agentic RAG</br>
Ref:
- [Rag Playground](https://github.com/zhangsquared/rag-playground)
- [User LlamaIndex to build agentic rag](https://github.com/zhangsquared/agentic-rag-playground)

## Set up env

```bash
uv sync
```

Env var: Create a `.env` file
```bash
GOOGLE_API_KEY=...
GITHUB_TOKEN=...
```

### First time set up env

```bash
uv init .
uv add ruff  # lint tool
```

Dependencies:

```bash
uv add sentence_transformers chromadb google-genai python-dotenv
uv add langchain langchain-google-genai langgraph
uv add fastapi uvicorn
```

## Code structure

```bash
raw-agentic-rag/
├── __init__.py # basic setting
├── ingest/
│   └── load_docs.py # batch job to chuck, embed documents, and save to storage
├── query/
│   ├── agent.py # AI Agent
│   ├── api.py # FastAPI, stateless online service
│   └── rag_query_engine.py # given a user query, perform vector retrieval,
│                           # apply reranking, and generate a response.
└── storage/ # persisted index document
```

Run the batch ingestion job:
```bash
uv run python -m ingest.load_docs
```

Local test rag query engine:
```bash
uv run python -m query.rag_query_engine
```

Local test LangChain agent:
```bash
uv run python -m query.agent
```

Start the online service:
```bash
uv run uvicorn query.api:app --host 0.0.0.0 --port 8000
```

Run lint tool:

```bash
ruff check --select I --fix . | uv run ruff format .
```

## Choose embedding and reranking model

Requirement: free, fast, no requirement for GPU

- Embedding: `BAAI/bge-small-en-v1.5`
- Reranking: `BAAI/bge-reranker-base`
- LLM Model: `gemini-2.5-flash-lite`

[config](./config/__init__.py)
