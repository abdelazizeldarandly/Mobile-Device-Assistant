from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseIndex:
    """Document embeddings computed once and reused for every query."""

    def __init__(self, documents: List[str]):
        self.documents = documents
        self._model = SentenceTransformer(MODEL_NAME)  # downloads on first run, then cached
        # Shape: (num_documents, embedding_dim). Normalized, so each row has length 1.
        self._doc_embeddings = self._model.encode(
            documents, normalize_embeddings=True
        )

    def search(self, query: str) -> List[int]:
        """Return ALL document indices, ordered from most to least similar."""
        query_embedding = self._model.encode(
            QUERY_INSTRUCTION + query, normalize_embeddings=True
        )
        # For normalized vectors, dot product == cosine similarity
        scores = self._doc_embeddings @ query_embedding
        return np.argsort(-scores, kind="stable").tolist()