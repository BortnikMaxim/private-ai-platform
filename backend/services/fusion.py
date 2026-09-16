"""Reciprocal Rank Fusion of several ranked candidate lists.

Why RRF rather than score blending
----------------------------------
Cosine similarity from the embedding model and Okapi BM25 live on different,
unbounded and query-dependent scales. Min-max normalising them into a weighted
sum invents a comparability that does not exist: the same BM25 score means
something different for a one-word query than for a ten-word one, and the
normalisation is dominated by whichever branch happened to return an outlier.

RRF only reads *positions*. Each list contributes ``1 / (k + rank)``, the
contributions are summed per document, and nothing about the underlying score
scales enters the result. It is the standard remedy for exactly this problem
(Cormack et al., 2009) and it needs no tuning beyond ``k``, which controls how
sharply the top of each list is favoured — a larger ``k`` flattens the curve and
lets agreement between branches matter more than a single branch's top hit.

The original scores are carried through untouched for diagnostics; they are
never mixed arithmetically.
"""

from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_RRF_K = 60

# Keys copied from whichever branch first contributed a chunk.
_PAYLOAD_KEYS = (
    "user_id",
    "document_id",
    "filename",
    "page",
    "chunk_index",
    "text",
)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Iterable[dict[str, Any]]],
    key: str = "point_id",
    rrf_k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked lists into one, ordered by descending RRF score.

    Each input list must already be ordered best-first. A chunk present in
    several lists is emitted once, with the diagnostics of every branch that
    found it and the sum of their reciprocal-rank contributions — so agreement
    between branches is what pushes a chunk up.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")

    fused: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            identity = item.get(key)

            if identity is None:
                raise ValueError(f"cannot fuse a candidate without '{key}'")

            identity = str(identity)
            entry = fused.get(identity)

            if entry is None:
                entry = {key: identity, "rrf_score": 0.0}

                for payload_key in _PAYLOAD_KEYS:
                    if payload_key in item:
                        entry[payload_key] = item[payload_key]

                fused[identity] = entry
                order.append(identity)
            else:
                # Fill gaps only; never overwrite what another branch supplied.
                for payload_key in _PAYLOAD_KEYS:
                    if entry.get(payload_key) is None and item.get(payload_key) is not None:
                        entry[payload_key] = item[payload_key]

            entry["rrf_score"] += 1.0 / (rrf_k + rank)

            # Per-branch diagnostics travel with the chunk, unmixed.
            for diagnostic in (
                "dense_score",
                "dense_rank",
                "lexical_score",
                "lexical_rank",
                "score",
                "vector_score",
            ):
                if diagnostic in item and item[diagnostic] is not None:
                    entry[diagnostic] = item[diagnostic]

    results = list(fused.values())

    for entry in results:
        entry["rrf_score"] = round(entry["rrf_score"], 8)

    # Stable ordering: RRF score first, then insertion order, so two chunks
    # with identical fused scores never swap places between runs.
    position = {identity: index for index, identity in enumerate(order)}
    results.sort(key=lambda entry: (-entry["rrf_score"], position[entry[key]]))

    return results[:limit] if limit is not None else results
