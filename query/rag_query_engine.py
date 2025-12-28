import os
from typing import List

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

from config import LLM_MODEL_NAME, RERANK_MODEL_NAME
from ingest.load_docs import load_index
from shared.embed import embed_chunk

RETRIVAL_TOP_K = 10
RERANK_TOP_N = 3


class QueryEngine:
    def __init__(self):
        # take environment variables from .env file
        load_dotenv()

        # Load the vector store index
        self.collection = load_index()

        # Initialize Google GenAI client
        self.google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _retrieve(self, query: str, top_k: int) -> List[str]:
        query_embedding = embed_chunk(query)
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        return results["documents"][0]

    def _rerank(self, query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
        cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
        pairs = [(query, chunk) for chunk in retrieved_chunks]
        scores = cross_encoder.predict(pairs)

        scored_chunks = list(zip(retrieved_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored_chunks][:top_k]

    def _generate_response(self, query: str, context_chunks: List[str]) -> str:
        prompt = f"""You are a knowledge assistant. Please generate an accurate answer based on the user's question and the following passages.

        User question: {query}

        Relevant context:
        {"\n\n".join(context_chunks)}

        Please answer based only on the content above and do not fabricate information.
        """
        response = self.google_client.models.generate_content(
            model=LLM_MODEL_NAME, contents=prompt
        )
        return response.text

    def query(self, user_query: str) -> str:
        retrieved_chunks = self._retrieve(user_query, RETRIVAL_TOP_K)
        for i, chunk in enumerate(retrieved_chunks):
            print(f"[retrived{i}] {chunk}\n")

        reranked_chunks = self._rerank(user_query, retrieved_chunks, RERANK_TOP_N)
        for i, chunk in enumerate(reranked_chunks):
            print(f"(Reranked)[{i}] {chunk}\n")

        response = self._generate_response(user_query, reranked_chunks)
        return response


# global query engine instance
query_engine = QueryEngine()

if __name__ == "__main__":
    user_querys = [
        "What is the responsibility of CNCF Governing Board?",
        "What is marketing committee?",
    ]
    for user_query in user_querys:
        response = query_engine.query(user_query)
        print("-------------------")
        print(user_query)
        print(response)
        print("-------------------")
