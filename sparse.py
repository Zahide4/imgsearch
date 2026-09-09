"""Client-side BM25, so the index can live somewhere other than Qdrant Cloud.

Both the crawler and the API used to hand Qdrant a `models.Document` and let
its inference service embed the text. That service exists only on the managed
cloud; a self-hosted instance rejects it. Since the 10M plan puts Qdrant on a
Hetzner box, the sparse half of hybrid retrieval has to be computed here.

The model name is the same one the cloud was using (`Qdrant/bm25`), so the term
ids hash identically and vectors written by either route are interchangeable.
The collection is configured with `Modifier.IDF`, which means Qdrant applies
inverse document frequency from its own corpus statistics -- the client owes it
term frequencies only, which is exactly what this produces.

Documents and queries are embedded differently, and it matters: a document
carries TF weights, a query is a flat list of terms at 1.0. Using `embed` for
queries would weight a repeated word in the query as if it were evidence.
"""
from functools import lru_cache

MODEL = 'Qdrant/bm25'


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so that merely importing this module does not pull
    # fastembed into processes that never build a sparse vector.
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(MODEL)


def _as_vector(embedding):
    from qdrant_client import models
    return models.SparseVector(indices=[int(i) for i in embedding.indices],
                               values=[float(v) for v in embedding.values])


def document(text):
    """Term frequencies for a corpus document."""
    return _as_vector(next(_model().embed([text or ''])))


def query(text):
    """Terms for a search, unweighted."""
    return _as_vector(next(_model().query_embed([text or ''])))


def documents(texts):
    """Batched, for ingest: one pass over the tokenizer instead of many."""
    return [_as_vector(e) for e in _model().embed(list(texts))]
