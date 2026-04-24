from __future__ import annotations

import json
import math
import os
import re
import time
import hashlib
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return max(1, len(encoding.encode(text)))
    except Exception:
        pass
    return max(1, math.ceil(len(tokenize(text)) * 1.3))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(slots=True)
class MemoryRecord:
    content: str
    memory_type: str
    priority: int
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.content)

    def matches(self, query: str) -> float:
        query_tokens = Counter(tokenize(query))
        content_tokens = Counter(tokenize(self.content + " " + " ".join(self.tags)))
        return cosine_similarity(query_tokens, content_tokens)


class MemoryBackend(Protocol):
    name: str
    backend_mode: str

    def add(self, record: MemoryRecord) -> None:
        ...

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        ...

    def all(self) -> list[MemoryRecord]:
        ...

    def clear(self) -> None:
        ...


def _record_to_json(record: MemoryRecord) -> dict[str, Any]:
    return asdict(record)


def _record_from_json(data: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        content=str(data["content"]),
        memory_type=str(data["memory_type"]),
        priority=int(data["priority"]),
        tags=list(data.get("tags", [])),
        metadata=dict(data.get("metadata", {})),
        created_at=float(data.get("created_at", time.time())),
    )


class ConversationBufferMemory:
    """Short-term memory that keeps the newest turns only."""

    name = "conversation_buffer"
    backend_mode = "memory"

    def __init__(self, max_turns: int = 12) -> None:
        self._records: deque[MemoryRecord] = deque(maxlen=max_turns)

    def add(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        scored = sorted(
            self._records,
            key=lambda record: (record.matches(query), record.created_at),
            reverse=True,
        )
        return scored[:limit]

    def all(self) -> list[MemoryRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()


class JsonStoreMixin:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_records(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as file:
            return [_record_from_json(item) for item in json.load(file)]

    def _write_records(self, records: list[MemoryRecord]) -> None:
        with self.path.open("w", encoding="utf-8") as file:
            json.dump([_record_to_json(record) for record in records], file, indent=2)


class RedisLongTermMemory(JsonStoreMixin):
    """Long-term key-value memory.

    Uses a real Redis list when redis-py and Redis are available. The JSON
    fallback keeps the lab deterministic in offline environments.
    """

    name = "redis_long_term"

    def __init__(
        self,
        path: Path = Path("data/redis_long_term.json"),
        redis_url: str | None = None,
        key: str = "lab17:redis_long_term",
        require_redis: bool | None = None,
    ) -> None:
        super().__init__(path)
        self.key = key
        self.backend_mode = "json_fallback"
        self._redis: Any | None = None
        self._require_redis = _env_enabled("LAB17_REQUIRE_REDIS") if require_redis is None else require_redis
        self._connect(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    def _connect(self, redis_url: str) -> None:
        if _env_enabled("LAB17_DISABLE_REDIS"):
            return
        try:
            import redis

            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
        except Exception as exc:
            if self._require_redis:
                raise RuntimeError(f"Redis backend is required but unavailable: {exc}") from exc
            return

        self._redis = client
        self.backend_mode = "redis"

    def add(self, record: MemoryRecord) -> None:
        if self._redis:
            self._redis.rpush(self.key, json.dumps(_record_to_json(record)))
            return

        records = self._read_records()
        records.append(record)
        self._write_records(records)

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        records = self.all()
        scored = sorted(
            records,
            key=lambda record: (record.matches(query), record.priority, record.created_at),
            reverse=True,
        )
        return scored[:limit]

    def all(self) -> list[MemoryRecord]:
        if self._redis:
            return [_record_from_json(json.loads(item)) for item in self._redis.lrange(self.key, 0, -1)]
        return self._read_records()

    def clear(self) -> None:
        if self._redis:
            self._redis.delete(self.key)
            return
        self._write_records([])


class JsonEpisodicLogMemory(JsonStoreMixin):
    """Append-only episodic log for user experiences and event traces."""

    name = "json_episodic_log"
    backend_mode = "json"

    def __init__(self, path: Path = Path("data/episodic_log.json")) -> None:
        super().__init__(path)

    def add(self, record: MemoryRecord) -> None:
        records = self._read_records()
        records.append(record)
        self._write_records(records)

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        scored = sorted(
            self._read_records(),
            key=lambda record: (record.matches(query), record.created_at),
            reverse=True,
        )
        return scored[:limit]

    def all(self) -> list[MemoryRecord]:
        return self._read_records()

    def clear(self) -> None:
        self._write_records([])


class ChromaSemanticMemory(JsonStoreMixin):
    """Semantic memory inspired by Chroma vector retrieval.

    Uses a real persistent Chroma collection when chromadb is installed. The
    local JSON fallback preserves the same retrieve-by-similar interface without
    external dependencies.
    """

    name = "chroma_semantic"

    def __init__(
        self,
        path: Path = Path("data/chroma_semantic.json"),
        persist_dir: Path | None = None,
        collection_name: str = "lab17_memory",
        require_chroma: bool | None = None,
    ) -> None:
        super().__init__(path)
        self.backend_mode = "json_fallback"
        self.collection_name = collection_name
        self.persist_dir = persist_dir or Path(os.getenv("CHROMA_PATH", "data/chroma_db"))
        self._client: Any | None = None
        self._collection: Any | None = None
        self._require_chroma = _env_enabled("LAB17_REQUIRE_CHROMA") if require_chroma is None else require_chroma
        self._connect()

    def _connect(self) -> None:
        if _env_enabled("LAB17_DISABLE_CHROMA"):
            return
        try:
            import chromadb

            chroma_host = os.getenv("CHROMA_HOST", "").strip()
            if chroma_host:
                client = chromadb.HttpClient(
                    host=chroma_host,
                    port=int(os.getenv("CHROMA_PORT", "8000")),
                )
            else:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(self.persist_dir))
            collection = client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:
            if self._require_chroma:
                raise RuntimeError(f"Chroma backend is required but unavailable: {exc}") from exc
            return

        self._client = client
        self._collection = collection
        self.backend_mode = "chroma"

    def add(self, record: MemoryRecord) -> None:
        if self._collection:
            self._collection.add(
                ids=[self._record_id(record)],
                documents=[record.content],
                embeddings=[_hashed_embedding(record.content)],
                metadatas=[
                    {
                        "memory_type": record.memory_type,
                        "priority": record.priority,
                        "tags": ",".join(record.tags),
                        "metadata_json": json.dumps(record.metadata),
                        "created_at": record.created_at,
                    }
                ],
            )
            return

        enriched = MemoryRecord(
            content=record.content,
            memory_type=record.memory_type,
            priority=record.priority,
            tags=record.tags,
            metadata={**record.metadata, "embedding": dict(Counter(tokenize(record.content)))},
            created_at=record.created_at,
        )
        records = self._read_records()
        records.append(enriched)
        self._write_records(records)

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        if self._collection:
            result = self._collection.query(
                query_embeddings=[_hashed_embedding(query)],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            return self._records_from_chroma_result(result)

        query_vec = Counter(tokenize(query))
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self._read_records():
            embedding = Counter(record.metadata.get("embedding", {}))
            score = cosine_similarity(query_vec, embedding)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (item[0], item[1].priority, item[1].created_at), reverse=True)
        return [record for _, record in scored[:limit]]

    def all(self) -> list[MemoryRecord]:
        if self._collection:
            result = self._collection.get(include=["documents", "metadatas"])
            return self._records_from_chroma_get(result)
        return self._read_records()

    def clear(self) -> None:
        if self._client:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            return
        self._write_records([])

    def _record_id(self, record: MemoryRecord) -> str:
        raw = f"{record.created_at}:{record.memory_type}:{record.content}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _records_from_chroma_result(self, result: dict[str, Any]) -> list[MemoryRecord]:
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            self._record_from_chroma(document, metadata)
            for document, metadata in zip(documents, metadatas, strict=False)
        ]

    def _records_from_chroma_get(self, result: dict[str, Any]) -> list[MemoryRecord]:
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            self._record_from_chroma(document, metadata)
            for document, metadata in zip(documents, metadatas, strict=False)
        ]

    def _record_from_chroma(self, document: str, metadata: dict[str, Any] | None) -> MemoryRecord:
        metadata = metadata or {}
        metadata_json = metadata.get("metadata_json") or "{}"
        try:
            extra_metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            extra_metadata = {}
        tags = [tag for tag in str(metadata.get("tags", "")).split(",") if tag]
        return MemoryRecord(
            content=document,
            memory_type=str(metadata.get("memory_type", "semantic")),
            priority=int(metadata.get("priority", 2)),
            tags=tags,
            metadata=extra_metadata,
            created_at=float(metadata.get("created_at", time.time())),
        )


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _hashed_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        index = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
