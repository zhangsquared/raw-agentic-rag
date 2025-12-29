# Raw Agentic RAG

Use LangChain to buid an agentic RAG

<img width="1488" height="520" alt="image" src="https://github.com/user-attachments/assets/cb7dd9fb-5b79-4500-9ea5-c2eca26f8dba" />

[original graph](https://docs.google.com/presentation/d/1K2VCMqOAlOVnsUab6jgWRqVD6KxNtc0bzXsOfQeWedM/edit?slide=id.p#slide=id.p)

Related git repo:
- [Rag Playground](https://github.com/zhangsquared/rag-playground)
- [User LlamaIndex to build agentic rag](https://github.com/zhangsquared/agentic-rag-playground)
- [Slack Bot](https://github.com/zhangsquared/zz-slack-bot)

## Set up env

```bash
uv sync
```

Env var: Create a `.env` file
```bash
GOOGLE_API_KEY=...
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
├── ingest/
│   └── load_docs.py # batch job to chuck, embed documents, and save to storage
├── query/
│   ├── agent.py # AI Agent
│   ├── api.py # FastAPI, stateless online service
│   └── rag_query_engine.py # given a user query, perform vector retrieval,
│                           # apply reranking, and generate a response. It is a (MCP) tool registered in AI Agent. 
└── storage/ # persisted index document
```

### Local development

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

### Run lint tool

```bash
ruff check --select I --fix . | uv run ruff format .
```

## Choose embedding and reranking model

Requirement: free, fast, no requirement for GPU

- Embedding: `BAAI/bge-small-en-v1.5`
- Reranking: `BAAI/bge-reranker-base`
- LLM Model: `gemini-2.5-flash-lite`

[config](./config/__init__.py)


## Future improvement

1. Gemini free tier has rate limiting

```bash
Error processing query: Error calling model 'models/gemini-2.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash-lite\nPlease retry in 18.380998879s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-lite', 'location': 'global'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '18s'}]}}
```

2. Model loading introduces a noticeable start-up delay during the first run

On the first run, `load_docs.py` or `rag_query_engine.py` may be slow because the embedding model and reranking model need to be downloaded.
In particular, the reranking model can take up to an hour to download on the first run.

3. Improve the doc loading and chunking method

4. How to incress Python FastAPI parallel processing request

By default, `uv run uvicorn query.api:app --host 0.0.0.0 --port 8000` starts a single worker process. Due to Python’s Global Interpreter Lock (GIL), a single process can only execute one thread of Python bytecode at a time.

Although running `uv run uvicorn query.api:app --host 0.0.0.0 --port 8000 --workers 4` will start four worker processes, the requests still appear to be handled sequentially rather than in parallel.

Possible bottlenecks include limited CPU or GPU resources, the use of a single shared database connection, or rate limiting imposed by the Google API.
