import os
import re
import csv
import json
import uuid
import shutil
import logging
from math import sqrt
from typing import List, Optional, Dict, Any, Tuple, Iterable

import pandas as pd
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Vector store + embeddings
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# =========================== env / logging ===========================
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("rag_mvp")


def _log(event: str, payload: Dict[str, Any], query_id: Optional[str] = None):
    rec = {"event": event, **({"query_id": query_id}
                              if query_id else {}), **payload}
    logger.info(json.dumps(rec, ensure_ascii=False))


def _indent(txt: str, pad: int = 2) -> str:
    pad_s = " " * pad
    return "\n".join(pad_s + line for line in (txt or "").splitlines())


def _text_preview(t: str, n: int = 240) -> str:
    t = (t or "").strip()
    return t[:n] + ("…" if len(t) > n else "")


def _vec_norm(v: List[float]) -> float:
    return float(sqrt(sum((x*x for x in v))))


# =========================== constants / paths ===========================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_DIR = os.path.join(ROOT, "agent/assets")
CSV_PATH = os.getenv("CSV_PATH", os.path.join(ASSETS_DIR, "sample-file.csv"))

CHROMA_DIR = os.path.join(ROOT, "chroma_data_mvp")
COLLECTION_NAME = "csv_collection_mvp"
EMB_MODEL = os.getenv(
    "EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EXPORT_DIR = os.path.join(ROOT, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() in ("1", "true", "yes")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_OK = USE_OPENAI and bool(os.getenv("OPENAI_API_KEY"))

# =========================== vector/index helpers ===========================
_embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)


def _ensure_chroma_client() -> chromadb.ClientAPI:
    settings = Settings(anonymized_telemetry=False, allow_reset=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=settings)
    try:
        client.heartbeat()
        _ = client.list_collections()
    except Exception as e:
        logger.warning(f"Chroma init issue; resetting: {e}")
        client.reset()
    return client


def _load_vs() -> Chroma:
    client = _ensure_chroma_client()
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=_embeddings, client=client)


# =========================== normalization ===========================
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    s = (s or "").strip().lower().replace("\n", " ").replace("\r", " ")
    return _SLUG_RE.sub("_", s).strip("_")


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not str(
        c).strip().lower().startswith("unnamed")]
    df = df[cols].copy()
    df.rename(columns={c: _slug(str(c)) for c in df.columns}, inplace=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)
    return df


def _row_to_text(row: pd.Series) -> str:
    lines = []
    for col, val in row.items():
        sval = "" if pd.isna(val) else str(val).strip()
        lines.append(f"{col}: {sval}")
    return "\n".join(lines)


def _norm_pet(val: str) -> Optional[bool]:
    v = (val or "").strip().lower()
    if not v or v in {"-", "na", "n/a"}:
        return None
    if v in {"yes", "true", "y", "1", "pet-friendly", "pet friendly"}:
        return True
    if "allowed" in v or "welcome" in v or "ok" in v or "okay" in v:
        if "no" not in v and "not" not in v:
            return True
    if v in {"no", "false", "n", "0"}:
        return False
    if "no pet" in v or "not pet" in v or "not allowed" in v:
        return False
    return None


def _extract_bhk(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*[-\s]?bhk", (text or "").lower())
    return int(m.group(1)) if m else None


# =========================== schema (tiny) ===========================
SCHEMA: Dict[str, Any] = {
    "fields": {},
    "name_slugs": [],
}


def _build_schema_from_docs(docs: List[Document]) -> Dict[str, Any]:
    fields: Dict[str, set] = {}
    name_slugs = set()
    for d in docs:
        m = d.metadata or {}
        for k, v in m.items():
            if v is None:
                continue
            fields.setdefault(k, set()).add(str(v))
        nslug = (m.get("name_slug") or "").strip()
        if nslug:
            name_slugs.add(nslug)
    return {
        "fields": {k: sorted(list(v)) for k, v in fields.items()},
        "name_slugs": sorted(list(name_slugs)),
    }


# =========================== price parsing ===========================
_PRICE_NUM = re.compile(r"(?i)\b(\d{1,3}(?:[,\s]\d{3})+|\d+)\s*(k)?\b")
_CURRENCY = re.compile(r"[₹$,]")


def _parse_price_inr(val: str) -> Optional[int]:
    s = (val or "").strip()
    if not s:
        return None
    s = _CURRENCY.sub("", s)
    m = _PRICE_NUM.search(s)
    if not m:
        return None
    num = m.group(1).replace(",", "").replace(" ", "")
    mult = 1000 if (m.group(2) or "").lower() == "k" else 1
    try:
        v = int(float(num) * mult)
        return v if 100 <= v <= 10_000_000 else None
    except:
        return None

# =========================== ingest ===========================


def rebuild_from_csv() -> dict:
    if not os.path.exists(CSV_PATH):
        raise HTTPException(
            status_code=404, detail=f"CSV not found at {CSV_PATH}")
    raw_df = pd.read_csv(CSV_PATH, dtype=str,
                         keep_default_na=False, na_filter=False)
    df = _normalize_df(raw_df)

    # known field keys
    name_key = "property_names" if "property_names" in df.columns else None
    prop_type_key = "property_type" if "property_type" in df.columns else None
    location_key = "location" if "location" in df.columns else None
    pet_key = "pet_friendly" if "pet_friendly" in df.columns else None
    bhk_key = "bhk" if "bhk" in df.columns else None

    # price candidates (optional)
    _price_candidates = {
        "price", "price_inr", "cost", "starting_price", "starting_price_inr",
        "startingprice", "startingpriceinr", "price_per_night", "price_pernight",
    }
    price_key = next((c for c in df.columns if c in _price_candidates), None)

    docs: List[Document] = []
    ids: List[str] = []
    for i, row in df.iterrows():
        content = _row_to_text(row)

        name_val = str(row.get(name_key, "") or "") if name_key else ""
        prop_type = str(row.get(prop_type_key, "")
                        or "") if prop_type_key else ""
        location = (str(row.get(location_key, "") or "")
                    if location_key else "").title() or None
        pf = _norm_pet(str(row.get(pet_key, "") or "")) if pet_key else None

        bhk_val = None
        bhk_col = str(row.get(bhk_key, "") or "") if bhk_key else ""
        if bhk_col:
            m = re.search(r"\d+", bhk_col)
            if m:
                try:
                    bhk_val = int(m.group(0))
                except:
                    bhk_val = None
        if bhk_val is None:
            bhk_val = _extract_bhk(prop_type)

        price_val = _parse_price_inr(
            str(row.get(price_key, "") or "")) if price_key else None

        name_slug = _slug(name_val) if name_val else None
        meta = {
            "row_index": int(i),
            "source_id": f"row-{i}",
            "property_name": name_val or None,
            "name_slug": name_slug,
            "location": location,
            "pet_friendly": pf,
            "bhk": bhk_val,
            "price_inr": price_val,
        }
        docs.append(Document(page_content=content, metadata=meta))
        ids.append(meta["source_id"])

    # fresh persistent client + add docs
    client = _ensure_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    vs = Chroma(collection_name=COLLECTION_NAME,
                embedding_function=_embeddings, client=client)
    vs.add_documents(docs, ids=ids)
    try:
        vs._client.persist()  # type: ignore
    except Exception:
        pass

    # embedding preview (small sample)
    sample = docs[:5]
    try:
        vecs = _embeddings.embed_documents([d.page_content for d in sample])
        dims = len(vecs[0]) if vecs else 0
        _log("rag_mvp:embeddings_finalized", {
            "embedding_model": EMB_MODEL,
            "dimension": dims,
            "samples": [
                {"id": d.metadata["source_id"], "text_preview": _text_preview(d.page_content, 160),
                 "first_dims": [float(x) for x in v[:12]], "l2_norm": round(_vec_norm(v), 6)}
                for d, v in zip(sample, vecs)
            ],
        })
    except Exception as e:
        _log("rag_mvp:embeddings_finalized", {
             "embedding_model": EMB_MODEL, "error": f"preview_failed: {e}"})

    # tiny schema
    global SCHEMA
    SCHEMA = _build_schema_from_docs(docs)

    return {
        "rows": int(df.shape[0]),
        "docs": int(len(docs)),
        "schema_fields": list(SCHEMA["fields"].keys()),
        "name_slugs": SCHEMA["name_slugs"][:10],
    }

# =========================== where helpers ===========================


def _to_chroma_where(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert filters into Chroma operator-where:
      {a:1} -> {"a":{"$eq":1}}
      {"price_inr_lte": 15000} -> {"price_inr":{"$lte":15000}}
      {"price_inr_gte": 5000}  -> {"price_inr":{"$gte":5000}}
      {a:1,b:2} -> {"$and":[...]}
    """
    if not filters:
        return None
    parts = []
    for k, v in filters.items():
        if v is None:
            continue
        if k.endswith("_lte"):
            parts.append({k[:-4]: {"$lte": v}})
        elif k.endswith("_gte"):
            parts.append({k[:-4]: {"$gte": v}})
        else:
            parts.append({k: {"$eq": v}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


def _flat_from_op(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not where:
        return None
    if "$and" in where:
        flat = {}
        for d in where["$and"]:
            k = next(iter(d.keys()))
            vv = d[k]
            if isinstance(vv, dict):
                if "$eq" in vv:
                    flat[k] = vv["$eq"]
                # Chroma 'get' accepts only equality; non-$eq are ignored in flat form.
            else:
                flat[k] = vv
        return flat or None
    k = next(iter(where.keys()))
    vv = where[k]
    return {k: vv.get("$eq")} if isinstance(vv, dict) and "$eq" in vv else None


def _contains_all_slug_tokens(meta: Dict[str, Any], requested_slug: str) -> bool:
    """
    Local check used by the last-resort (R5-lite) step:
    accept a candidate only if all tokens from the requested slug
    (split by underscores/spaces) appear in the candidate's slug or name.
    """
    if not requested_slug:
        return False
    tokens = [t for t in re.split(r"[_\s]+", requested_slug.strip().lower()) if t]
    if not tokens:
        return False
    cand_slug = (meta.get("name_slug") or meta.get("slug") or "").strip().lower()
    cand_name = (meta.get("property_name") or meta.get("name") or "").strip().lower()
    hay = f"{cand_slug} {cand_name}"
    return all(tok in hay for tok in tokens)


def retrieve_with_relaxation(
    vs: Chroma,
    query: str,
    top_k: int,
    filters: Dict[str, Any],
    query_id: Optional[str] = None,
    min_score_last_resort: float = 0.80,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    R0: strict filters (current behavior)
    R1: drop 'location' (keep 'name_slug') if strict returns 0 hits
    R5-lite: unfiltered vector; locally keep only candidates that contain all slug tokens
             (guarded by a minimum score). Then truncate to top_k.
    """
    # --- R0: strict ---
    hits, contexts, diag = retrieve(vs, query, top_k, filters, query_id=query_id)
    if len(hits) > 0:
        diag["rung"] = "R0"
        return hits, contexts, diag

    # --- R1: drop location if both name_slug & location were present ---
    requested_slug = (filters.get("name_slug") or "").strip()
    requested_loc = (filters.get("location") or "").strip()
    if requested_slug and requested_loc:
        relaxed_filters = dict(filters)
        relaxed_filters.pop("location", None)
        r1_hits, r1_ctx, r1_diag = retrieve(vs, query, top_k, relaxed_filters, query_id=query_id)
        if len(r1_hits) > 0:
            r1_diag["rung"] = "R1"
            return r1_hits, r1_ctx, r1_diag

    # --- R5-lite: unfiltered vector, then local re-filter by slug tokens & score ---
    # Only attempt if we have a requested slug; otherwise the last resort may be too noisy.
    if requested_slug:
        try:
            pairs = vs.similarity_search_with_score(query, k=max(20, top_k), filter=None)
        except Exception:
            pairs = []

        kept: List[Dict[str, Any]] = []
        kept_contexts: List[str] = []

        for doc, score in pairs:
            meta = dict(doc.metadata or {})
            if float(score) < min_score_last_resort:
                continue
            if not _contains_all_slug_tokens(meta, requested_slug):
                continue
            txt = doc.page_content or ""
            kept.append({
                "id": meta.get("source_id"),
                "row_index": meta.get("row_index"),
                "score": float(score),
                "meta": {
                    "location": meta.get("location"),
                    "bhk": meta.get("bhk"),
                    "pet_friendly": meta.get("pet_friendly"),
                    "name_slug": meta.get("name_slug"),
                    "price_inr": meta.get("price_inr"),
                },
                "text_preview": _text_preview(txt, 360),
                "text_len": len(txt),
                "full_text": txt,
            })
            kept_contexts.append(txt)

        kept.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        if kept:
            diag2 = {
                "query": query,
                "filters": filters,
                "where_op": None,
                "where_flat": None,
                "top_k": top_k,
                "hit_count": len(kept),
                "retrieval_mode": "qa_vector_last_resort",
                "rung": "R5-lite",
            }
            return kept[:top_k], kept_contexts[:top_k], diag2

    # Fall back to the original diag (R0) but mark rung=fallback for clarity
    diag["rung"] = "fallback"
    return [], [], diag

# =========================== parsing ===========================
NEG_PETS = re.compile(r"\b(?:non[-\s]?pet|no\s+pets?|not\s+pet)\b", re.I)
POS_PETS = re.compile(
    r"\b(?:pet[-\s]?friendly|pets?\s*(?:allowed|ok|okay|welcome|permitted))\b", re.I)
PRICE_UNDER = re.compile(
    r"(?i)\b(?:under|below|less\s*than|<=?|≤|up\s*to)\s*(?:₹|\brs\.?\b|\$)?\s*"
    r"(\d{1,3}(?:[,\s]\d{3})+|\d+)\s*(k)?\b"
)
_CURRENCY = re.compile(r"(?:[₹$,]|\brs\.?\b)", re.I)

def _to_inr(n: str, k: Optional[str]) -> Optional[int]:
    try:
        base = int(n.replace(",", "").replace(" ", ""))
        return base * 1000 if (k or "").lower() == "k" else base
    except:
        return None


def _is_enumerate_query(q: str) -> bool:
    ql = (q or "").lower()
    return bool(re.search(r"\blist\s+all\b", ql) or re.search(r"\bshow\s+all\b", ql) or ql.strip().startswith("all "))


def parse_query(query: str, schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    q = (query or "").strip()
    ql = q.lower()

    # intents: count / min_price / enumerate / detail / list
    if re.search(r"\bhow\s+many\b", ql):
        intent = "count"
    elif re.search(r"\b(cheapest|lowest\s+price|least\s+expensive)\b", ql):
        intent = "min_price"
    elif _is_enumerate_query(ql):
        intent = "enumerate"
    elif re.search(r"\b(details?|info|full)\b", ql):
        intent = "detail"
    else:
        intent = "list"

    slots: Dict[str, Any] = {}

    # bhk
    m_bhk = re.search(r"\b(\d+)\s*[-\s]?bhk\b", ql)
    if m_bhk:
        try:
            slots["bhk"] = int(m_bhk.group(1))
        except:
            pass

    # price under
    m_price = PRICE_UNDER.search(ql)
    if not m_price:
        # fallback: strip currency symbols/commas like "₹", "$", "Rs"
        ql_sanitized = _CURRENCY.sub("", ql)
        m_price = PRICE_UNDER.search(ql_sanitized)

    if m_price:
        n, kflag = m_price.group(1), m_price.group(2)
        px = _to_inr(n, kflag)
        if px:
            slots["price_inr_lte"] = px

    # pet-friendly
    if NEG_PETS.search(ql):
        slots["pet_friendly"] = False
    elif POS_PETS.search(ql):
        slots["pet_friendly"] = True

    # name slug
    for nslug in schema.get("name_slugs", []):
        if nslug and nslug.replace("_", " ") in ql:
            slots["name_slug"] = nslug
            intent = "detail"
            break

    # location (contains)
    for loc in schema.get("fields", {}).get("location", []):
        if loc and loc.lower() in ql:
            slots["location"] = loc
            break

    return intent, {k: v for k, v in slots.items() if v is not None}


# =========================== LLM wrapper ===========================
SYSTEM_PROMPT = (
    "You are a CSV-grounded assistant. Use ONLY the provided row contexts.\n"
    "- If no row clearly answers, reply exactly: 'Not enough information in CSV context.'\n"
    "- Prefer concise bullets for lists; one short paragraph + bullets for details.\n"
    "- Do not invent values not present in contexts.\n"
    "- Respect applied filters; never list items excluded by filters.\n"
)


def call_llm(messages: List[Dict[str, str]]) -> str:
    if not OPENAI_OK:
        user = next((m for m in messages if m["role"] == "user"), {})
        ctx_text = user.get("content", "")
        rows = [s for s in ctx_text.split("\n") if s.strip().startswith(
            ("property_names:", "location:", "bhk:", "pet_friendly:", "price_inr:"))]
        if not rows:
            return "Not enough information in CSV context."
        return "Answer derived from CSV rows:\n- " + "\n- ".join(rows[:10])
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, temperature=0.1)
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"OpenAI error: {e}")
        return "Not enough information in CSV context."

# =========================== API models ===========================


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 5
    rebuild: bool = False

    # enumeration / export controls
    page: int = 1
    page_size: int = 100
    export_all: bool = False
    export_format: Optional[str] = Field(default=None, description="csv|jsonl")
    export_fields: Optional[List[str]] = None
    sort_by: Optional[str] = Field(
        default=None, description="property_name|location|bhk|pet_friendly|price_inr|row_index")
    sort_dir: Optional[str] = Field(default="asc", description="asc|desc")

    # debug returns
    return_contexts: bool = True
    return_llm_prompt: bool = True
    return_llm_answer_raw: bool = True


class Hit(BaseModel):
    score: float
    row_index: Optional[int] = None
    id: Optional[str] = None
    preview: str


class RagResponse(BaseModel):
    rebuilt: bool
    query: str
    top_k: int
    hits: List[Hit]
    answer: Optional[str]
    debug: Dict[str, Any]

# =========================== metadata enumeration helpers ===========================
def _iter_metas(col, where_op: Optional[Dict[str, Any]], batch: int = 1000) -> Iterable[Dict[str, Any]]:
    def _make_eq_where_from_flat(flat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not flat:
            return None
        if len(flat) == 1:
            k, v = next(iter(flat.items()))
            return {k: {"$eq": v}}
        return {"$and": [{k: {"$eq": v}} for k, v in flat.items()]}

    # Normalize: None if empty/invalid composite
    where: Optional[Dict[str, Any]] = None
    if where_op:
        # Keep as-is; validator requires a single top-level operator or a single field
        where = where_op

    offset = 0
    while True:
        try:
            got = col.get(where=where, include=["metadatas"], limit=batch, offset=offset)  # type: ignore
        except Exception:
            # Fallback to equality-only filter if composite ops cause validation issues
            eq_where = _make_eq_where_from_flat(_flat_from_op(where_op))
            got = col.get(where=eq_where, include=["metadatas"], limit=batch, offset=offset)  # type: ignore

        metas = got.get("metadatas") or []
        ids = got.get("ids") or []
        if not metas:
            break

        for i, m in enumerate(metas):
            mm = dict(m or {})
            mm["_id"] = ids[i] if i < len(ids) else None
            yield mm

        if len(metas) < batch:
            break
        offset += batch

def _count_metas(col, where_op: Optional[Dict[str, Any]]) -> int:
    return sum(1 for _ in _iter_metas(col, where_op, batch=2000))


def _collect_metas(col, where_op: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(_iter_metas(col, where_op, batch=2000))


def _export_metas(col, where_op: Optional[Dict[str, Any]], fields: Optional[List[str]], fmt: str) -> Tuple[str, int]:
    wanted = fields or ["property_name", "location", "bhk",
                        "pet_friendly", "price_inr", "name_slug", "row_index", "source_id"]
    ts = uuid.uuid4().hex[:8]
    path = os.path.join(
        EXPORT_DIR, f"export_{ts}.{'csv' if fmt == 'csv' else 'jsonl'}")

    total = 0
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=wanted)
            w.writeheader()
            for m in _iter_metas(col, where_op, batch=2000):
                w.writerow({k: m.get(k) for k in wanted})
                total += 1
    else:
        with open(path, "w", encoding="utf-8") as f:
            for m in _iter_metas(col, where_op, batch=2000):
                obj = {k: m.get(k) for k in wanted}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += 1
    return path, total


def _sort_key(meta: Dict[str, Any], key: str):
    val = meta.get(key)
    # ensure stable, None goes last
    return (val is None, val)


def _format_price(v: Optional[int]) -> str:
    if v is None:
        return "—"
    s = f"{v:,}"
    return f"₹{s}"

# =========================== retrieval (QA path) ===========================


def retrieve(vs: Chroma, query: str, top_k: int, filters: Dict[str, Any], query_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    where_op = _to_chroma_where(filters)
    where_flat = _flat_from_op(where_op)
    k = max(1, min(top_k, 20))
    try:
        pairs = vs.similarity_search_with_score(query, k=k, filter=where_op)
    except Exception as e:
        _log("rag_mvp:retrieval_error", {"reason": str(
            e), "where_op": where_op, "fallback": "flat"}, query_id)
        try:
            pairs = vs.similarity_search_with_score(
                query, k=k, filter=where_flat)
        except Exception as e2:
            _log("rag_mvp:retrieval_error", {"reason": str(
                e2), "where_flat": where_flat, "fallback": "none"}, query_id)
            pairs = vs.similarity_search_with_score(query, k=k, filter=None)

    hits: List[Dict[str, Any]] = []
    contexts: List[str] = []
    for doc, score in pairs:
        meta = dict(doc.metadata or {})
        txt = doc.page_content or ""
        hits.append({
            "id": meta.get("source_id"),
            "row_index": meta.get("row_index"),
            "score": float(score),
            "meta": {
                "location": meta.get("location"),
                "bhk": meta.get("bhk"),
                "pet_friendly": meta.get("pet_friendly"),
                "name_slug": meta.get("name_slug"),
                "price_inr": meta.get("price_inr"),
            },
            "text_preview": _text_preview(txt, 360),
            "text_len": len(txt),
            "full_text": txt,
        })
        contexts.append(txt)

    diag = {
        "query": query,
        "filters": filters,
        "where_op": where_op,
        "where_flat": where_flat,
        "top_k": k,
        "hit_count": len(hits),
        "retrieval_mode": "qa_vector",
    }
    return hits, contexts, diag


def build_messages(query: str, filters: Dict[str, Any], contexts: List[str]) -> List[Dict[str, str]]:
    joined = "\n\n---\n\n".join(contexts)
    user = (
        f"Question:\n{query}\n\n"
        f"Applied filters: {json.dumps(filters, ensure_ascii=False)}\n"
        f"TopK: {len(contexts)}\n\n"
        f"Contexts:\n{joined}\n\n"
        f"Answer using only the contexts."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def log_retrieval(diag: Dict[str, Any], hits: List[Dict[str, Any]], query_id: Optional[str] = None):
    _log("rag_mvp:retrieval", {
        **diag,
        "hits": [
            {
                "id": h.get("id"),
                "row_index": h.get("row_index"),
                "score": round(float(h.get("score", 0.0)), 4),
                "meta": h.get("meta"),
                "preview": h.get("text_preview"),
                "text_len": h.get("text_len"),
            }
            for h in hits
        ],
        "context_char_count": sum(h.get("text_len", 0) for h in hits),
    }, query_id)


def log_llm_input(messages: List[Dict[str, str]], query_id: Optional[str] = None):
    contexts = []
    if messages and len(messages) >= 2:
        user = messages[-1].get("content", "")
        cx_start = user.find("Contexts:\n")
        if cx_start >= 0:
            ctx = user[cx_start + len("Contexts:\n"):].strip()
            contexts_txt = ctx[:4000] + ("…" if len(ctx) > 4000 else "")
            contexts = contexts_txt.split("\n\n---\n\n")
    _log("rag_mvp:llm_input", {
        "system": messages[0].get("content", "")[:600] if messages else "",
        "user_prefix": _text_preview(messages[-1].get("content", "") if messages else "", 600),
        "message_count": len(messages),
    }, query_id)
    if contexts:
        _log("rag_mvp:contexts_attached", {
            "chunks": [{"idx": i, "preview": _text_preview(c, 320), "char_len": len(c)} for i, c in enumerate(contexts)]
        }, query_id)


def log_llm_output(answer: str, model_name: str, query_id: Optional[str] = None):
    _log("rag_mvp:llm_output", {"model": model_name, "answer_preview": _text_preview(
        answer, 600), "answer_len": len(answer)}, query_id)

# =========================== human-readable audit ===========================


def _embed_for_hits(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"model": EMB_MODEL, "dimension": 0, "items": []}
    try:
        texts = [h.get("full_text") or "" for h in hits]
        if not texts:
            return out
        vecs = _embeddings.embed_documents(texts)
        dim = len(vecs[0]) if vecs else 0
        out["dimension"] = dim
        for h, v in zip(hits, vecs):
            out["items"].append({
                "id": h.get("id"),
                "row_index": h.get("row_index"),
                "first_dims": [float(x) for x in v[:8]],
                "l2_norm": round(_vec_norm(v), 6),
            })
    except Exception as e:
        out = {"model": EMB_MODEL, "error": f"embed_failed: {e}", "items": []}
    return out


def _human_block(title: str, body: str) -> str:
    bar = "=" * len(title)
    return f"{title}\n{bar}\n{body}\n"


def log_human_audit(query_id: str, sections: Dict[str, str]):
    block = f"[audit] query_id={query_id}\n" + \
        "".join(_human_block(k, _indent(v)) for k, v in sections.items())
    logger.info(block)


# =========================== router ===========================
router = APIRouter()


@router.post("/rag_agent", response_model=RagResponse)
def rag_agent(req: RagRequest = Body(...)):
    qid = uuid.uuid4().hex[:8]
    rebuilt = False
    if req.rebuild:
        rebuilt = True
        info = rebuild_from_csv()
        _log("rag_mvp:rebuild", {"info": info}, qid)

    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(
            status_code=400, detail="No index yet. Call with rebuild=true first.")

    vs = _load_vs()
    col = vs._collection

    # refresh schema cheaply
    try:
        got = col.get(include=["metadatas"], limit=10000)  # type: ignore
        metas = got.get("metadatas") or []
        docs = [Document(page_content="", metadata=m or {}) for m in metas]
        global SCHEMA
        SCHEMA = _build_schema_from_docs(docs)
    except Exception:
        pass

    # ---- parse ----
    intent, filters = parse_query(req.query, SCHEMA)
    where_op = _to_chroma_where(filters)

    # ---- planner ----
    # map intents to execution paths
    # count -> metadata count only
    # min_price -> scan matches, pick min price_inr
    # enumerate -> metadata listing with paging/export; no LLM
    # else -> QA vector path
    if intent == "count":
        total = _count_metas(col, where_op)
        _log("rag_mvp:enumeration_count", {
             "query": req.query, "filters": filters, "where": _flat_from_op(where_op), "total": total}, qid)

        answer = f"{total}"
        hits: List[Hit] = []
        debug = {
            "query_id": qid,
            "intent": intent,
            "filters_used": filters,
            "where_op": where_op,
            "total": total,
            "retrieval_mode": "enumerate_count",
        }
        log_human_audit(qid, {
            "Query": req.query,
            "Filters": json.dumps(filters, ensure_ascii=False, indent=2),
            "Retrieval Mode": "enumerate_count",
            "Count Result": str(total),
        })
        return RagResponse(rebuilt=rebuilt, query=req.query, top_k=req.top_k, hits=hits, answer=answer, debug=debug)

    if intent == "min_price":
        # collect all metas; filter to rows with price_inr
        metas = _collect_metas(col, where_op)
        priced = [m for m in metas if isinstance(
            m.get("price_inr"), (int, float))]
        priced.sort(key=lambda m: (m.get("price_inr")
                    is None, m.get("price_inr")))
        best = priced[0] if priced else None

        if not best:
            answer = "Not enough information in CSV context."
            log_human_audit(qid, {
                "Query": req.query,
                "Filters": json.dumps(filters, ensure_ascii=False, indent=2),
                "Retrieval Mode": "min_price_scan",
                "Result": "no row with price_inr",
            })
            return RagResponse(rebuilt=rebuilt, query=req.query, top_k=req.top_k, hits=[], answer=answer, debug={
                "query_id": qid, "intent": intent, "filters_used": filters, "where_op": where_op,
                "retrieval_mode": "min_price_scan", "reason": "no priced rows"
            })

        # build a concise answer; no LLM needed
        answer = (
            f"Cheapest match:\n"
            f"- {best.get('property_name') or best.get('name_slug') or best.get('source_id')} "
            f"({best.get('location')}, {best.get('bhk')} BHK)\n"
            f"- Pet-friendly: {'Yes' if best.get('pet_friendly') else 'No' if best.get('pet_friendly') is not None else '—'}\n"
            f"- Price: {_format_price(best.get('price_inr'))}"
        )

        _log("rag_mvp:min_price", {"query": req.query, "filters": filters, "where": _flat_from_op(where_op),
                                   "best_row": {k: best.get(k) for k in ("source_id", "row_index", "property_name", "location", "bhk", "pet_friendly", "price_inr", "name_slug")}}, qid)

        log_human_audit(qid, {
            "Query": req.query,
            "Filters": json.dumps(filters, ensure_ascii=False, indent=2),
            "Retrieval Mode": "min_price_scan",
            "Cheapest Row": json.dumps({k: best.get(k) for k in ("source_id", "row_index", "property_name", "location", "bhk", "pet_friendly", "price_inr", "name_slug")}, ensure_ascii=False, indent=2),
            "Answer": answer,
        })

        return RagResponse(rebuilt=rebuilt, query=req.query, top_k=req.top_k, hits=[], answer=answer, debug={
            "query_id": qid, "intent": intent, "filters_used": filters, "where_op": where_op,
            "retrieval_mode": "min_price_scan", "row": best
        })

    if intent == "enumerate" or req.export_all:
        # export or page
        retrieval_mode = "enumerate_list"
        if req.export_all:
            fmt = (req.export_format or "csv").lower()
            if fmt not in ("csv", "jsonl"):
                fmt = "csv"
            path, total = _export_metas(col, where_op, req.export_fields, fmt)
            _log("rag_mvp:enumeration_export", {"query": req.query, "filters": filters, "where": _flat_from_op(where_op),
                                                "export_format": fmt, "export_path": path, "total": total}, qid)
            answer = f"Exported {total} rows to {path}"
            log_human_audit(qid, {
                "Query": req.query,
                "Filters": json.dumps(filters, ensure_ascii=False, indent=2),
                "Retrieval Mode": "enumerate_export",
                "Export": f"path={path} total={total}",
            })
            return RagResponse(rebuilt=rebuilt, query=req.query, top_k=req.top_k, hits=[], answer=answer, debug={
                "query_id": qid, "intent": "enumerate_export", "filters_used": filters, "where_op": where_op,
                "retrieval_mode": "enumerate_export", "export_path": path, "total": total
            })

        # list page
        metas = _collect_metas(col, where_op)
        total = len(metas)

        sort_by = (req.sort_by or "property_name").strip()
        sort_dir = (req.sort_dir or "asc").lower()
        if sort_by not in {"property_name", "location", "bhk", "pet_friendly", "price_inr", "row_index"}:
            sort_by = "property_name"
        metas.sort(key=lambda m: _sort_key(m, sort_by),
                   reverse=(sort_dir == "desc"))

        page = max(1, req.page)
        page_size = max(1, min(req.page_size, 1000))
        start = (page-1)*page_size
        end = start + page_size
        page_items = metas[start:end]

        # build answer text (exhaustive page)
        lines = [
            f"Total matches: {total}. Page {page} (items {start+1}–{min(end, total)}). Sort: {sort_by} {sort_dir}."]
        for m in page_items:
            lines.append(
                f"- {m.get('property_name') or m.get('name_slug') or m.get('source_id')} | "
                f"{m.get('location') or '—'} | {m.get('bhk') or '—'} BHK | "
                f"Pet-friendly: {'Yes' if m.get('pet_friendly') else 'No' if m.get('pet_friendly') is not None else '—'} | "
                f"Price: {_format_price(m.get('price_inr'))}"
            )
        answer = "\n".join(lines)

        # hits (for API symmetry; using preview from metadata only)
        resp_hits: List[Hit] = [
            Hit(score=0.0, row_index=m.get("row_index"), id=m.get("source_id"),
                preview=f"property_name: {m.get('property_name')}\nlocation: {m.get('location')}\nbhk: {m.get('bhk')}\npet_friendly: {m.get('pet_friendly')}\nprice_inr: {m.get('price_inr')}")
            for m in page_items
        ]

        _log("rag_mvp:enumeration_page", {"query": req.query, "filters": filters, "where": _flat_from_op(where_op),
                                          "total": total, "page": page, "page_size": page_size,
                                          "sort_by": sort_by, "sort_dir": sort_dir,
                                          "items": [{k: m.get(k) for k in ("source_id", "row_index", "property_name", "location", "bhk", "pet_friendly", "price_inr", "name_slug")} for m in page_items]}, qid)

        log_human_audit(qid, {
            "Query": req.query,
            "Filters": json.dumps(filters, ensure_ascii=False, indent=2),
            "Retrieval Mode": "enumerate_list",
            "Page Summary": f"total={total} page={page} size={page_size} sort={sort_by} {sort_dir}",
            "First Items": "\n".join([f"{i+1}. {(m.get('property_name') or m.get('name_slug') or m.get('source_id'))} | {m.get('location')} | {m.get('bhk')} BHK | pet={m.get('pet_friendly')} | price={_format_price(m.get('price_inr'))}" for i, m in enumerate(page_items[:20])]) or "(none)",
        })

        return RagResponse(
            rebuilt=rebuilt, query=req.query, top_k=req.top_k, hits=resp_hits, answer=answer,
            debug={"query_id": qid, "intent": "enumerate", "filters_used": filters, "where_op": where_op,
                   "retrieval_mode": retrieval_mode, "total": total, "page": page, "page_size": page_size,
                   "sort_by": sort_by, "sort_dir": sort_dir}
        )

    # ---- QA (vector) path ----
    hits, contexts, diag = retrieve_with_relaxation(
        vs, req.query, req.top_k, filters, query_id=qid)
    log_retrieval(diag, hits, qid)
    messages = build_messages(req.query, filters, contexts)
    log_llm_input(messages, query_id=qid)
    answer = call_llm(messages)
    log_llm_output(
        answer, OPENAI_MODEL if OPENAI_OK else "local-fallback", query_id=qid)

    # embed preview + audit
    emb_info = _embed_for_hits(hits)
    sections = {
        "Query": req.query,
        "Filters": json.dumps(filters, ensure_ascii=False, indent=2),
        "Retrieval (final hits)": "\n".join(
            [f"[{i}] id={h.get('id')} row={h.get('row_index')} score={round(h.get('score', 0.0), 4)} "
             f"bhk={h.get('meta', {}).get('bhk')} pet={h.get('meta', {}).get('pet_friendly')} "
             f"loc={h.get('meta', {}).get('location')} slug={h.get('meta', {}).get('name_slug')}"
             for i, h in enumerate(hits)]
        ) or "(none)",
        "Context sent to LLM (per chunk, truncated)": "\n\n".join([f"--- Context #{i} ---\n{_text_preview(h.get('full_text') or '', 1600)}" for i, h in enumerate(hits)]) or "(no context)",
        "Embeddings (retrieved)": "model={0} dim={1}\n".format(emb_info.get("model"), emb_info.get("dimension")) + "\n".join(
            [f"- id={it.get('id')} row={it.get('row_index')} norm={it.get('l2_norm')} first_dims={it.get('first_dims')}" for it in emb_info.get("items", [])]
        ),
        "LLM Prompt (system + user)": "SYSTEM:\n{0}\n\nUSER:\n{1}".format(messages[0]["content"][:400], messages[-1]["content"][:600]),
        "LLM Final Answer": answer,
    }
    log_human_audit(qid, sections)

    resp_hits: List[Hit] = [
        Hit(score=h["score"], row_index=h["row_index"],
            id=h["id"], preview=h["text_preview"])
        for h in hits
    ]
    debug: Dict[str, Any] = {
        "query_id": qid, "intent": intent, "filters_used": filters, "use_openai": OPENAI_OK,
        "embedding_model": EMB_MODEL, "where_op": diag.get("where_op"), "where_flat": diag.get("where_flat"),
        "retrieval_mode": "qa_vector"
    }
    if req.return_contexts:
        debug["contexts"] = contexts
    if req.return_llm_prompt:
        debug["llm_prompt"] = messages
    if req.return_llm_answer_raw:
        debug["llm_answer_raw"] = answer

    return RagResponse(rebuilt=rebuilt, query=req.query, top_k=req.top_k, hits=resp_hits, answer=answer, debug=debug)
