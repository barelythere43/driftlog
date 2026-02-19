"""Reciprocal Rank Fusion to merge dense and sparse search results."""
import logging

logger = logging.getLogger(__name__)


def fuse_results(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 20,
) -> list[dict]:
    """Merge dense and sparse results by RRF. Each input dict has chunk_id, content, document_id, metadata."""
    # 1-based rank and dict by chunk_id (no dependency on similarity_score / score)
    dense_rank: dict[str, int] = {}
    dense_by_id: dict[str, dict] = {}
    for rank, d in enumerate(dense_results, start=1):
        cid = d["chunk_id"]
        dense_rank[cid] = rank
        dense_by_id[cid] = d

    sparse_rank: dict[str, int] = {}
    sparse_by_id: dict[str, dict] = {}
    for rank, s in enumerate(sparse_results, start=1):
        cid = s["chunk_id"]
        sparse_rank[cid] = rank
        sparse_by_id[cid] = s

    all_chunk_ids = set(dense_rank) | set(sparse_rank)
    dense_only = sum(1 for cid in all_chunk_ids if cid in dense_rank and cid not in sparse_rank)
    sparse_only = sum(1 for cid in all_chunk_ids if cid in sparse_rank and cid not in dense_rank)
    both = sum(1 for cid in all_chunk_ids if cid in dense_rank and cid in sparse_rank)
    logger.info(
        "fusion stats: dense_only=%d, sparse_only=%d, both=%d",
        dense_only,
        sparse_only,
        both,
    )

    fused: list[dict] = []
    for cid in all_chunk_ids:
        rrf_score = 0.0
        sources: list[str] = []
        if cid in dense_rank:
            rrf_score += 1.0 / (k + dense_rank[cid])
            sources.append("dense")
        if cid in sparse_rank:
            rrf_score += 1.0 / (k + sparse_rank[cid])
            sources.append("sparse")

        # Use content/metadata from whichever list had it ranked higher (lower rank)
        use_dense = cid in dense_rank and (
            cid not in sparse_rank or dense_rank[cid] <= sparse_rank[cid]
        )
        src = dense_by_id[cid] if use_dense else sparse_by_id[cid]
        fused.append(
            {
                "chunk_id": cid,
                "content": src["content"],
                "document_id": src["document_id"],
                "metadata": src["metadata"],
                "rrf_score": rrf_score,
                "sources": sources,
            }
        )

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused[:top_k]
