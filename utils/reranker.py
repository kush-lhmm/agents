from typing import List, Optional, Tuple

from utils.logger import logger

try:
    from sentence_transformers import CrossEncoder  # needs: pip install sentence-transformers
    _HAS_CE = True
except Exception:
    _HAS_CE = False

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if not _HAS_CE:
        logger.warning("sentence-transformers not installed; skipping reranker.")
        return None
    try:
        return CrossEncoder(model_name, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load CrossEncoder: {e}")
        return None

def rerank_pairs(
    ce,
    query: str,
    pairs: List[Tuple[str, float, Optional[int]]],
    top_n: int = 6,
) -> List[Tuple[str, float, Optional[int]]]:
    """
    Input pairs are (text, orig_score, row_index). Returns top_n reranked by CE score desc.
    """
    if not ce or not pairs:
        return pairs[:top_n]
    inputs = [(query, t) for (t, _, _) in pairs]
    scores = ce.predict(inputs)  # higher is better
    rescored = list(zip(pairs, scores))
    rescored.sort(key=lambda x: float(x[1]), reverse=True)
    out: List[Tuple[str, float, Optional[int]]] = []
    for (t, orig, ridx), _s in rescored[:top_n]:
        # keep original distance as 'score' for transparency
        out.append((t, orig, ridx))
    return out
