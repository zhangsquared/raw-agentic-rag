import os
from typing import List

import chromadb

from config import COLLECTION_NAME
from shared.embed import embed_chunk

STORAGE = os.path.abspath("./storage")


def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, "r") as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]


def load_and_chunk_docs() -> List[tuple[str, List[float]]]:
    chunks = split_into_chunks("doc.md")
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    return list(zip(chunks, embeddings))


def save_index(chucks_with_embeddings: List[tuple[str, List[float]]]) -> None:
    chroma_client = chromadb.PersistentClient(
        path=STORAGE,
    )
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    for i, (chunk, embedding) in enumerate(chucks_with_embeddings):
        chroma_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)],
        )
    print(f"Index and metadata successfully persisted to {STORAGE}.")


def load_index() -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(
        path=STORAGE,
    )
    chroma_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Index successfully loaded from chroma {chroma_collection.count()} vectors.")
    return chroma_collection


if __name__ == "__main__":
    embedings = load_and_chunk_docs()
    save_index(embedings)

    collection = load_index()
    print(f"Index loaded {collection}")
    for i, chunk in enumerate(collection.get()["documents"]):
        print(f"[{i}] {chunk}\n")
