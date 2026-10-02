from collections import defaultdict
from typing import List, Tuple

from app.data.devices import DEVICE_DOCUMENTS 
from app.utils.bm25_index import BM25Index
from app.utils.embeddings import DenseIndex

RRF_K = 60

# Built ONCE when this module is first imported, then reused for every request.
_bm25_index = BM25Index(DEVICE_DOCUMENTS)
_dense_index = DenseIndex(DEVICE_DOCUMENTS)


def reciprocal_rank_fusion(
    rankings: List[List[int]], k: int = RRF_K
) -> List[Tuple[int, float]]:
    """Fuse several ranked lists of document indices using RRF.

    Each ranking is a list of document indices, best first. A document gets
    1 / (k + rank) from every list, with rank starting at 1, and the
    contributions are summed.

    Returns (doc_index, fused_score) pairs, sorted from best to worst.
    """
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):  # rank 1 = best
            scores[doc_idx] += 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Search the device catalog with BM25 + dense retrieval fused by RRF.

    Returns the top-k (document_text, fused_rrf_score) tuples, best first.
    """
    sparse_ranking = _bm25_index.search(query)   # best first
    dense_ranking = _dense_index.search(query)   

    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking], k=RRF_K)

    return [(DEVICE_DOCUMENTS[idx], score) for idx, score in fused[:k]]
    