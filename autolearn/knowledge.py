"""Knowledge bases (retrieval-augmented generation): customers upload documents, the API retrieves the
most relevant passages (BM25 ranking via SQLite FTS5) and the LLM answers grounded in them, with citations.

    POST   /v1/knowledge/{kb}/documents      add documents (chunked automatically)
    GET    /v1/knowledge/{kb}/documents      list documents
    DELETE /v1/knowledge/{kb}/documents/{id} delete one document
    DELETE /v1/knowledge/{kb}                delete a whole knowledge base
    POST   /v1/knowledge/{kb}/search         ranked passages for a query

Search is hybrid when an embedding model is installed: BM25 keyword ranking and embedding (meaning) ranking
are merged with reciprocal rank fusion, so "how do I get my money back" finds the refund policy even though
it shares no words with it. Without an embedding model it falls back to BM25 alone.
    POST   /v1/chat/completions {"knowledge_base": "<kb>", ...}   grounded chat (see llm/routes.py)
"""

from __future__ import annotations

import re
import time
import uuid

import numpy as np

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from .config import PLANS, PROJECT_NAME_RE, validate_project_name

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_documents (
  tenant TEXT NOT NULL, kb TEXT NOT NULL, doc_id TEXT NOT NULL, title TEXT, chars INTEGER NOT NULL,
  chunks INTEGER NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (tenant, kb, doc_id)
);
CREATE TABLE IF NOT EXISTS kb_vectors (
  tenant TEXT NOT NULL, kb TEXT NOT NULL, doc_id TEXT NOT NULL, chunk INTEGER NOT NULL, vec BLOB NOT NULL,
  PRIMARY KEY (tenant, kb, doc_id, chunk)
);
CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks USING fts5(
  text, tenant UNINDEXED, kb UNINDEXED, doc_id UNINDEXED, title UNINDEXED, chunk UNINDEXED,
  tokenize = 'porter unicode61'
);
"""

WORD = re.compile(r"[A-Za-z0-9]+")
STOP = set("a an and are as at be by for from has have how i in is it of on or that the this to was what when "
           "where which who why will with you your do does did can could should would".split())


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split on paragraph/sentence boundaries into ~size-char chunks with overlap."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = cur[-overlap:] + " " + p
        else:
            cur = f"{cur} {p}" if cur else p
        while len(cur) > size * 1.5:  # very long sentence: hard split
            chunks.append(cur[:size].strip())
            cur = cur[size - overlap:]
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def fts_query(q: str) -> str | None:
    terms = [t.lower() for t in WORD.findall(q) if t.lower() not in STOP and len(t) > 1]
    if not terms:
        terms = [t.lower() for t in WORD.findall(q)]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms)) or None


RRF_K = 60  # reciprocal rank fusion constant (Cormack et al., 2009)


class Knowledge:
    def __init__(self, db, embedder=None):
        self.db = db
        self.embedder = embedder or (lambda: None)  # callable returning an Embedder or None
        with db.lock:
            db.conn.executescript(SCHEMA)

    # ------------------------------------------------------------- vectors --
    def _embed_missing(self, tenant, kb, limit=5000):
        """Embed chunks that have no vector yet (new uploads, or documents added before embeddings existed)."""
        emb = self.embedder()
        if emb is None:
            return 0
        rows = self.db.q(
            "SELECT c.doc_id, c.chunk, c.text FROM kb_chunks c WHERE c.tenant=? AND c.kb=? AND NOT EXISTS "
            "(SELECT 1 FROM kb_vectors v WHERE v.tenant=c.tenant AND v.kb=c.kb AND v.doc_id=c.doc_id "
            "AND v.chunk=c.chunk) LIMIT ?", (tenant, kb, limit))
        if not rows:
            return 0
        vecs = emb.encode([r["text"] for r in rows]).astype(np.float16)
        self.db.many("INSERT OR REPLACE INTO kb_vectors VALUES (?,?,?,?,?)",
                     [(tenant, kb, r["doc_id"], int(r["chunk"]), v.tobytes()) for r, v in zip(rows, vecs)])
        return len(rows)

    def _semantic(self, tenant, kb, query, n):
        emb = self.embedder()
        if emb is None:
            return []
        self._embed_missing(tenant, kb)
        rows = self.db.q("SELECT doc_id, chunk, vec FROM kb_vectors WHERE tenant=? AND kb=?", (tenant, kb))
        if not rows:
            return []
        m = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float16).reshape(len(rows), -1)
        q = emb.encode([query], query=True)[0].astype(np.float32)
        sims = m.astype(np.float32) @ q
        top = np.argsort(-sims)[:n]
        return [(rows[i]["doc_id"], int(rows[i]["chunk"]), float(sims[i])) for i in top]

    def add(self, tenant, kb, docs):
        out = []
        now = time.time()
        for d in docs:
            doc_id = d.get("id") or "doc_" + uuid.uuid4().hex[:12]
            self.delete_doc(tenant, kb, doc_id)  # re-upload replaces
            chunks = chunk_text(d["text"])
            with self.db.lock:
                self.db.conn.execute("BEGIN")
                try:
                    self.db.conn.executemany(
                        "INSERT INTO kb_chunks(text, tenant, kb, doc_id, title, chunk) VALUES (?,?,?,?,?,?)",
                        [(c, tenant, kb, doc_id, d.get("title"), i) for i, c in enumerate(chunks)])
                    self.db.conn.execute("INSERT INTO kb_documents VALUES (?,?,?,?,?,?,?)",
                                         (tenant, kb, doc_id, d.get("title"), len(d["text"]), len(chunks), now))
                    self.db.conn.execute("COMMIT")
                except Exception:
                    self.db.conn.execute("ROLLBACK")
                    raise
            out.append({"id": doc_id, "title": d.get("title"), "chunks": len(chunks)})
        self._embed_missing(tenant, kb)
        return out

    def delete_doc(self, tenant, kb, doc_id):
        with self.db.lock:
            self.db.conn.execute("DELETE FROM kb_chunks WHERE tenant=? AND kb=? AND doc_id=?", (tenant, kb, doc_id))
            self.db.conn.execute("DELETE FROM kb_vectors WHERE tenant=? AND kb=? AND doc_id=?", (tenant, kb, doc_id))
            return self.db.conn.execute("DELETE FROM kb_documents WHERE tenant=? AND kb=? AND doc_id=?",
                                        (tenant, kb, doc_id)).rowcount

    def delete_kb(self, tenant, kb):
        with self.db.lock:
            self.db.conn.execute("DELETE FROM kb_chunks WHERE tenant=? AND kb=?", (tenant, kb))
            self.db.conn.execute("DELETE FROM kb_vectors WHERE tenant=? AND kb=?", (tenant, kb))
            return self.db.conn.execute("DELETE FROM kb_documents WHERE tenant=? AND kb=?", (tenant, kb)).rowcount

    def docs(self, tenant, kb):
        return self.db.q("SELECT doc_id AS id, title, chars, chunks, created_at FROM kb_documents "
                         "WHERE tenant=? AND kb=? ORDER BY created_at", (tenant, kb))

    def list_kbs(self, tenant):
        return self.db.q("SELECT kb AS name, COUNT(*) documents, SUM(chunks) chunks, MAX(created_at) updated_at "
                         "FROM kb_documents WHERE tenant=? GROUP BY kb ORDER BY kb", (tenant,))

    def chunk_count(self, tenant):
        return self.db.one("SELECT COALESCE(SUM(chunks),0) n FROM kb_documents WHERE tenant=?", (tenant,))["n"]

    def search(self, tenant, kb, query, top_k=5):
        n = max(20, top_k * 4)
        fq = fts_query(query)
        kw = self.db.q(
            "SELECT doc_id, chunk, bm25(kb_chunks) AS score FROM kb_chunks "
            "WHERE kb_chunks MATCH ? AND tenant=? AND kb=? ORDER BY score LIMIT ?", (fq, tenant, kb, n)) if fq else []
        sem = self._semantic(tenant, kb, query, n)
        if not sem:  # keyword-only mode: BM25 score (lower is better -> negate)
            fused = [((r["doc_id"], int(r["chunk"])), -r["score"], "keyword") for r in kw[:top_k]]
        else:
            scores, how = {}, {}
            for rank, r in enumerate(kw):
                k = (r["doc_id"], int(r["chunk"]))
                scores[k] = scores.get(k, 0) + 1 / (RRF_K + rank + 1)
                how[k] = "keyword"
            for rank, (d, c, _) in enumerate(sem):
                k = (d, c)
                scores[k] = scores.get(k, 0) + 1 / (RRF_K + rank + 1)
                how[k] = "both" if how.get(k) == "keyword" else "semantic"
            fused = [(k, s, how[k]) for k, s in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]
        out = []
        for (doc_id, chunk), score, match in fused:
            r = self.db.one("SELECT text, title FROM kb_chunks WHERE tenant=? AND kb=? AND doc_id=? AND chunk=?",
                            (tenant, kb, doc_id, chunk))
            if r:
                out.append({"document_id": doc_id, "title": r["title"], "chunk": chunk, "text": r["text"],
                            "score": round(score, 4), "match": match})
        return out


def grounding_message(passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {p['title'] or p['document_id']}: {p['text']}" for i, p in enumerate(passages))
    return ("Answer using ONLY the numbered sources below. Cite sources like [1]. If the answer is not in the "
            f"sources, say you don't know.\n\nSources:\n{ctx}")


class DocIn(BaseModel):
    id: str | None = Field(default=None, max_length=128)
    title: str | None = Field(default=None, max_length=500)
    text: str = Field(min_length=1, max_length=2_000_000)


class DocsIn(BaseModel):
    documents: list[DocIn] = Field(min_length=1, max_length=1000)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


def register_knowledge_routes(app, db, tenant_any, tenant_admin, embedder=None):
    kbs = Knowledge(db, embedder)
    app.state.knowledge = kbs

    def check_name(kb):
        if not validate_project_name(kb):
            raise HTTPException(422, f"knowledge base name must match {PROJECT_NAME_RE}")

    @app.get("/v1/knowledge", tags=["knowledge"])
    def list_kbs(p=Depends(tenant_any)):
        return kbs.list_kbs(p.tenant)

    @app.post("/v1/knowledge/{kb}/documents", tags=["knowledge"], status_code=201)
    def add_docs(kb: str, body: DocsIn, p=Depends(tenant_admin)):
        check_name(kb)
        limit = PLANS[p.plan]["max_rows"]
        est = sum(max(1, len(d.text) // 750) for d in body.documents)
        if limit is not None and kbs.chunk_count(p.tenant) + db.count_tenant_samples(p.tenant) + est > limit:
            raise HTTPException(402, f"storage limit of {limit} rows/chunks reached for plan '{p.plan}'")
        out = kbs.add(p.tenant, kb, [d.model_dump() for d in body.documents])
        db.audit(p.tenant, p.key_id, "kb.add", {"kb": kb, "documents": len(out)})
        return {"knowledge_base": kb, "documents": out}

    @app.get("/v1/knowledge/{kb}/documents", tags=["knowledge"])
    def list_docs(kb: str, p=Depends(tenant_any)):
        check_name(kb)
        return kbs.docs(p.tenant, kb)

    @app.delete("/v1/knowledge/{kb}/documents/{doc_id}", tags=["knowledge"])
    def delete_doc(kb: str, doc_id: str, p=Depends(tenant_admin)):
        if not kbs.delete_doc(p.tenant, kb, doc_id):
            raise HTTPException(404, "document not found")
        db.audit(p.tenant, p.key_id, "kb.delete_document", {"kb": kb, "id": doc_id})
        return {"deleted": doc_id}

    @app.delete("/v1/knowledge/{kb}", tags=["knowledge"])
    def delete_kb(kb: str, p=Depends(tenant_admin)):
        n = kbs.delete_kb(p.tenant, kb)
        db.audit(p.tenant, p.key_id, "kb.delete", {"kb": kb, "documents": n})
        return {"deleted": kb, "documents": n}

    @app.post("/v1/knowledge/{kb}/search", tags=["knowledge"])
    def search(kb: str, body: SearchIn, p=Depends(tenant_any)):
        check_name(kb)
        db.add_usage(p.tenant, "kb_searches", 1)
        return {"knowledge_base": kb, "results": kbs.search(p.tenant, kb, body.query, body.top_k)}

    return kbs
