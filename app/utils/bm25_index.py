from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Lowercase and split on whitespace, as required by the assignment."""
    return text.lower().split()


class BM25Index:

    def __init__(self, documents: List[str]):
        self.documents = documents
        tokenized_corpus = [tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)  # the expensive part, done once

    def search(self, query: str) -> List[int]:
        """Return ALL document indices, ordered from most to least relevant."""
        scores = self._bm25.get_scores(tokenize(query))
        # Sort by descending score; a stable sort keeps the original order on ties
        return np.argsort(-scores, kind="stable").tolist()


        