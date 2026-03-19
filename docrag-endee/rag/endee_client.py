"""
Endee Vector Database — HTTP API Client
Fully verified from Endee C++ source (main.cpp + msgpack_ndd.hpp).
 
Key facts from source:
- INSERT: meta and filter must be JSON-encoded STRINGS (C++ calls .s() on them)
  meta is stored as raw UTF-8 bytes: vec.meta.assign(meta_str.begin(), meta_str.end())
- SEARCH: response is MessagePack, Content-Type: application/msgpack
  ResultSet struct: MSGPACK_DEFINE(results)        → packed as 1-element array
  VectorResult struct: MSGPACK_DEFINE(similarity, id, meta, filter, norm, vector)
  So raw decode = [[sim, id, meta_bytes, filter, norm, vector], ...]
- LIST INDEXES: returns {"indexes": [{name: "...", ...}]}
- CREATE: returns plain text "Index created successfully"
- INSERT: returns HTTP 200/400 with no body
"""
 
from __future__ import annotations
import os, uuid, json, requests
from typing import Any
 
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("[EndeeClient] WARNING: msgpack not installed. Run: pip install msgpack")
 
 
class EndeeClient:
    def __init__(self, host=None, api_key=None):
        self.base_url = (host or os.getenv("ENDEE_HOST", "http://localhost:8080")).rstrip("/")
        self.api_key  = api_key or os.getenv("ENDEE_API_KEY", "")
        self.session  = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": self.api_key})
 
    def create_index(self, name, dimension, metric="cosine", quantization="FLOAT32"):
        """POST /api/v1/index/create. 409 = already exists, treat as OK."""
        resp = self.session.post(
            f"{self.base_url}/api/v1/index/create",
            json={"index_name": name, "dim": dimension, "space_type": metric},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 409:
            print(f"[EndeeClient] Index '{name}' already exists — skipping.")
            return {"status": "already_exists"}
        resp.raise_for_status()
        return {"status": "ok"}
 
    def index_exists(self, name):
        """GET /api/v1/index/list → {"indexes": [{name, ...}]}"""
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/index/list")
            if resp.status_code != 200:
                return False
            data = resp.json()
            indexes = data.get("indexes", []) if isinstance(data, dict) else data
            return any(
                (isinstance(i, dict) and i.get("name") == name) or i == name
                for i in indexes
            )
        except Exception:
            return False
 
    def list_indexes(self):
        resp = self.session.get(f"{self.base_url}/api/v1/index/list")
        resp.raise_for_status()
        return resp.json()
 
    def health(self):
        resp = self.session.get(f"{self.base_url}/api/v1/health")
        resp.raise_for_status()
        return resp.json()
 
    def upsert(self, index_name, vectors):
        """
        POST /api/v1/index/{index_name}/vector/insert
        
        CRITICAL from C++ source:
          vec.meta.assign(meta_str.begin(), meta_str.end())  — meta is a JSON STRING
          vec.filter = std::string(item["filter"].s())       — filter is a JSON STRING
        Both must be passed as JSON-encoded strings, not objects.
        """
        endee_vectors = []
        for v in vectors:
            raw_meta = v.get("metadata", v.get("meta", {}))
            clean_meta = {
                k: (val[:2000] if isinstance(val, str) else val)
                for k, val in raw_meta.items()
                if isinstance(val, (str, int, float, bool))
            }
 
            entry = {
                "id":     v["id"],
                "vector": v.get("values", v.get("vector", [])),
                "meta":   json.dumps(clean_meta),  # JSON STRING, not object
            }
 
            raw_filter = v.get("filter", {})
            if raw_filter:
                entry["filter"] = json.dumps(raw_filter)  # JSON STRING, not object
 
            endee_vectors.append(entry)
 
        resp = self.session.post(
            f"{self.base_url}/api/v1/index/{index_name}/vector/insert",
            json=endee_vectors,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return {"status": "ok"}
 
    def search(self, index_name, query_vector, top_k=5, include_metadata=True, filters=None):
        """
        POST /api/v1/index/{index_name}/search
        
        Response is MessagePack (application/msgpack).
        
        ResultSet struct (MSGPACK_DEFINE(results)):
          Packed as 1-element array: [ [result1, result2, ...] ]
        
        VectorResult struct (MSGPACK_DEFINE(similarity, id, meta, filter, norm, vector)):
          Packed as 6-element array: [similarity, id, meta_bytes, filter, norm, vector]
          
        meta_bytes are raw UTF-8 bytes of the JSON string we stored.
        """
        if not HAS_MSGPACK:
            raise RuntimeError("msgpack not installed. Run: pip install msgpack")
 
        payload = {"vector": query_vector, "k": top_k, "include_vectors": False}

        if filters:
            f = filters if isinstance(filters, list) else [filters]
            payload["filter"] = json.dumps(f)
 
        resp = self.session.post(
            f"{self.base_url}/api/v1/index/{index_name}/search",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
 
        # Decode MessagePack
        # raw = [ [result1, result2, ...] ]  (ResultSet has 1 field: results)
        raw = msgpack.unpackb(resp.content, raw=False)
 
        # Extract the results list from ResultSet wrapper
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            result_list = raw[0]   # unwrap ResultSet
        elif isinstance(raw, (list, tuple)):
            result_list = raw      # already a flat list
        else:
            result_list = []
 
        results = []
        for r in result_list:
            if isinstance(r, (list, tuple)) and len(r) >= 3:
                # VectorResult: [similarity, id, meta_bytes, filter, norm, vector]
                similarity  = float(r[0]) if r[0] is not None else 0.0
                vec_id      = r[1] if r[1] is not None else ""
                meta_bytes  = r[2]
 
                # Decode meta bytes back to dict
                try:
                    if isinstance(meta_bytes, (bytes, bytearray)):
                        meta_str = meta_bytes.decode("utf-8", errors="ignore")
                    else:
                        meta_str = str(meta_bytes) if meta_bytes else "{}"
                    meta = json.loads(meta_str) if meta_str.strip() else {}
                except Exception:
                    meta = {}
 
            elif isinstance(r, dict):
                # Fallback if msgpack decoded as dict (shouldn't happen but be safe)
                similarity = float(r.get("similarity", r.get("score", 0.0)))
                vec_id     = r.get("id", "")
                meta_raw   = r.get("meta", b"")
                if isinstance(meta_raw, (bytes, bytearray)):
                    meta_raw = meta_raw.decode("utf-8", errors="ignore")
                try:
                    meta = json.loads(meta_raw) if meta_raw else {}
                except Exception:
                    meta = {}
            else:
                continue
 
            results.append({
                "id":       vec_id,
                "score":    similarity,
                "metadata": meta,
            })
 
        return results
 
    @staticmethod
    def make_id():
        return str(uuid.uuid4())