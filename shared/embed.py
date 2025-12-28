from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL_NAME

embedding_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()
