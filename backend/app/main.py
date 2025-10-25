from fastapi import FastAPI, HTTPException
import logging
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Any, Optional
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import weaviate
import hashlib
import os
import time

# Config
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_CLASS = "HTMLChunk"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_TOKENS = 500

app = FastAPI(title="HTML Chunk Search")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow CORS for local frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    url: HttpUrl
    query: str


class Match(BaseModel):
    chunk_id: int
    text: str
    html: str
    # score: higher is more relevant (we convert Weaviate distance -> similarity)
    score: float
    # raw distance returned by Weaviate when available (lower = more similar)
    distance: Optional[float] = None


def get_weaviate_client():
    """
    Construct a weaviate.Client. Different weaviate-client releases accept
    different constructor signatures; try several forms and raise a clear
    error if none succeed.
    """
    last_exc = None

    if hasattr(weaviate, "WeaviateClient"):
        try:
            return weaviate.WeaviateClient(url=WEAVIATE_URL)
        except Exception as e:
            last_exc = e
        try:
            return weaviate.WeaviateClient(WEAVIATE_URL)
        except Exception as e:
            last_exc = e

    # v3: weaviate.Client
    if hasattr(weaviate, "Client"):
        try:
            return weaviate.Client(url=WEAVIATE_URL)
        except Exception as e:
            last_exc = e
        try:
            return weaviate.Client(WEAVIATE_URL)
        except Exception as e:
            last_exc = e
        try:
            return weaviate.Client()
        except Exception as e:
            last_exc = e

    raise RuntimeError(
        "Unable to construct a Weaviate client using detected library API.\n"
        f"Last error: {last_exc}\n\n"
    )


def ensure_schema(client: weaviate.Client):
    # Create class if not exists
    if not client.schema.exists(WEAVIATE_CLASS):
        class_obj = {
            "class": WEAVIATE_CLASS,
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "html", "dataType": ["text"]},
                {"name": "url", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["int"]},
            ],
        }
        client.schema.create_class(class_obj)


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


def html_to_chunks(html: str, max_tokens: int = CHUNK_TOKENS):
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts and styles
    for s in soup(["script", "style", "noscript"]):
        s.decompose()

    body = soup.body or soup

    # Collect block-level text nodes (p, li, div, h1..h6)
    blocks = []
    selectors = ["p", "li", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"]
    for tag in body.find_all(selectors):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            blocks.append((text, str(tag)))

    # Fallback: if no blocks, use all text
    if not blocks:
        txt = body.get_text(separator=" ", strip=True)
        if txt:
            blocks = [(txt, str(body))]

    chunks = []
    current_tokens = []
    current_htmls = []
    current_text_tokens = 0

    def flush_chunk():
        if not current_tokens:
            return None
        text = " ".join(current_tokens)
        html_snippet = "\n".join(current_htmls)
        current_tokens.clear()
        current_htmls.clear()
        return text, html_snippet

    for idx, (text, html_snippet) in enumerate(blocks):
        tok_len = len(tokenizer.encode(text))
        # If single block > max_tokens, break it by sentence approx (naive)
        if tok_len > max_tokens:
            # split by sentences (simple period split)
            parts = [p.strip() for p in text.split(".") if p.strip()]
            for p in parts:
                plen = len(tokenizer.encode(p))
                if current_text_tokens + plen <= max_tokens:
                    current_tokens.append(p)
                    current_htmls.append(p)
                    current_text_tokens += plen
                else:
                    flushed = flush_chunk()
                    if flushed:
                        chunks.append(flushed)
                    current_text_tokens = len(tokenizer.encode(p))
                    current_tokens.append(p)
                    current_htmls.append(p)
        else:
            if current_text_tokens + tok_len <= max_tokens:
                current_tokens.append(text)
                current_htmls.append(html_snippet)
                current_text_tokens += tok_len
            else:
                flushed = flush_chunk()
                if flushed:
                    chunks.append(flushed)
                current_tokens.append(text)
                current_htmls.append(html_snippet)
                current_text_tokens = tok_len

    # final flush
    final = flush_chunk()
    if final:
        chunks.append(final)

    # Compose chunk dicts
    result = []
    for i, (t, h) in enumerate(chunks):
        result.append({"chunk_id": i, "text": t, "html": h})
    return result


def index_chunks(client: weaviate.Client, url: str, chunks: List[dict]):
    # Index into Weaviate. This function requires a valid weaviate client.
    if not client:
        raise RuntimeError("Weaviate client is not available. Ensure the weaviate-client package is installed and compatible, and WEAVIATE_URL points to a running instance.")

    ensure_schema(client)
    # upsert each chunk with embedding
    for ch in chunks:
        emb = embedder.encode(ch["text"]).tolist()
        properties = {"text": ch["text"], "html": ch["html"], "url": url, "chunk_id": ch["chunk_id"]}
        # Use a deterministic UUID based on URL + chunk_id so indexing is idempotent
        obj_id = hashlib.sha1(f"{url}:{ch['chunk_id']}".encode("utf-8")).hexdigest()
        try:
            # Try to create the object with the deterministic id.
            client.data_object.create(data_object=properties, class_name=WEAVIATE_CLASS, uuid=obj_id, vector=emb)
        except Exception as e:
            # If the object already exists or create fails, attempt an update to ensure properties/vector are current.
            try:
                client.data_object.update(data_object=properties, class_name=WEAVIATE_CLASS, uuid=obj_id)
                # Some clients support updating the vector via a separate call; if supported, attempt it.
                try:
                    client.data_object.replace(data_object=properties, class_name=WEAVIATE_CLASS, uuid=obj_id, vector=emb)
                except Exception:
                    # best-effort: ignore if replace not supported by client version
                    pass
            except Exception:
                # If update also fails, log and continue (don't break the whole indexing)
                logger.debug("Failed to create or update object %s: %s", obj_id, e)


def extract_items_from_weaviate_response(res: Any) -> List[dict]:
    """Recursively search the Weaviate response for objects that look like
    query result items (contain 'properties' or expected fields) and return them.
    This makes the parser robust to small differences between client versions.
    """
    found: List[dict] = []

    def visit(obj: Any):
        if isinstance(obj, dict):
            # If this dict looks like a result item, capture it
            if "properties" in obj or ("text" in obj and "html" in obj):
                found.append(obj)
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(res)
    return found


@app.post("/search", response_model=List[Match])
def search(req: SearchRequest):
    try:
        resp = requests.get(str(req.url), timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    html = resp.text
    chunks = html_to_chunks(html)

    client = get_weaviate_client()

    # Index chunks
    try:
        index_chunks(client, str(req.url), chunks)
    except Exception as e:
        # If weaviate not ready, return error with hint
        raise HTTPException(status_code=500, detail=f"Failed to index chunks into Weaviate: {e}. Is Weaviate running at {WEAVIATE_URL}?")

    # Compute query embedding and search using Weaviate.
    # We compute the query embedding once and keep a numpy copy so we can
    # recompute local cosine similarity scores as a fallback (and to ensure
    # the API always returns a meaningful score).
    q_vec = embedder.encode(req.query)
    try:
        q_vec_list = q_vec.tolist()
    except Exception:
        q_vec_list = list(q_vec)
    qv = np.array(q_vec, dtype=float)
    matches: List[Match] = []

    # Use the same client we used for indexing. get_weaviate_client() will
    # raise a RuntimeError with guidance if the client cannot be constructed.
    try:
        # client variable was created earlier for indexing; reuse it here
        res = (
            client.query
            .get(WEAVIATE_CLASS, ["text", "html", "url", "chunk_id"]) 
            .with_near_vector({"vector": q_vec_list})
            .with_additional(["distance"])
            .with_limit(10)
            .do()
        )
        logger.info("Weaviate raw response: %s", res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weaviate query failed: {e}")

    # Parse response robustly
    try:
        items = extract_items_from_weaviate_response(res)
        logger.info("Weaviate parsed items count=%d", len(items))
        for item in items:
            # item may be a dict with 'properties' or already the properties
            props = item.get("properties") if isinstance(item.get("properties"), dict) else item

            # Prefer Weaviate-provided distance if present; otherwise compute
            # cosine similarity locally between the query vector and the
            # returned text's embedding so the API always returns a score.
            distance = None
            score = 0.0
            # check both '_additional' and 'additional' keys
            add = None
            if isinstance(item.get("_additional"), dict):
                add = item.get("_additional")
            elif isinstance(item.get("additional"), dict):
                add = item.get("additional")

            if isinstance(add, dict) and add.get("distance") is not None:
                distance = float(add.get("distance"))
                # convert distance (lower better) into similarity score (0..1]
                try:
                    score = 1.0 / (1.0 + distance)
                except Exception:
                    score = float(distance)
            elif isinstance(item.get("additional"), list) and item.get("additional"):
                try:
                    d = item["additional"][0].get("distance", None)
                    if d is not None:
                        distance = float(d)
                        score = 1.0 / (1.0 + distance)
                except Exception:
                    distance = None
            elif item.get("distance") is not None:
                distance = float(item.get("distance"))
                score = 1.0 / (1.0 + distance)
            else:
                # Compute local cosine similarity as fallback
                try:
                    text_val = props.get("text", "")
                    if text_val:
                        ev = embedder.encode(text_val)
                        evn = np.array(ev, dtype=float)
                        denom = (np.linalg.norm(qv) * np.linalg.norm(evn))
                        score = float(np.dot(qv, evn) / denom) if denom > 0 else 0.0
                except Exception:
                    score = 0.0

            matches.append(Match(chunk_id=props.get("chunk_id", -1), text=props.get("text", ""), html=props.get("html", ""), score=score, distance=distance))
    except Exception as e:
        logger.exception("Failed to parse Weaviate response: %s", e)

    # Deduplicate matches by text (fall back to html) while preserving order.
    deduped: List[Match] = []
    seen: set = set()
    for m in matches:
        key = (m.text or "").strip()
        if not key:
            key = (m.html or "").strip()
        # normalize key to avoid tiny differences
        key_norm = " ".join(key.split())[:1000]
        if key_norm in seen:
            continue
        seen.add(key_norm)
        deduped.append(m)

    # Return up to 10 results (Weaviate asked for 10 but dedupe may reduce them)
    return deduped[:10]
