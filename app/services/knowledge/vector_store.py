"""Milvus adapter for dense knowledge vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VectorRow:
    chunk_id: int
    vector: list[float]


@dataclass(frozen=True)
class VectorHit:
    chunk_id: int
    score: float


class MilvusKnowledgeStore:
    def __init__(
        self,
        uri: str | None = None,
        collection_name: str = "knowledge",
        dimension: int = 1024,
        *,
        client: Any | None = None,
    ):
        self.collection_name = collection_name
        self.dimension = dimension
        if client is None:
            if not uri:
                raise ValueError("Milvus URI is required")
            from pymilvus import MilvusClient

            client = MilvusClient(uri=uri)
        self.client = client

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(collection_name=self.collection_name)
            return
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("chunk_id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection(collection_name=self.collection_name)

    def upsert(self, rows: list[VectorRow]) -> list[int]:
        if not rows:
            return []
        for row in rows:
            if len(row.vector) != self.dimension:
                raise ValueError(f"Milvus vectors must have {self.dimension} dimensions")
        response = self.client.upsert(
            collection_name=self.collection_name,
            data=[{"chunk_id": row.chunk_id, "embedding": row.vector} for row in rows],
        )
        ids = list(response.get("ids", []))
        expected_ids = [row.chunk_id for row in rows]
        if ids != expected_ids:
            raise RuntimeError("Milvus returned IDs that do not match knowledge chunk primary keys")
        return ids

    def search(self, vector: list[float], limit: int) -> list[VectorHit]:
        if len(vector) != self.dimension:
            raise ValueError(f"Milvus query vector must have {self.dimension} dimensions")
        if limit < 1:
            return []
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=limit,
            anns_field="embedding",
            output_fields=[],
            search_params={"metric_type": "COSINE", "params": {}},
        )
        return [VectorHit(int(hit["id"]), float(hit["distance"])) for hit in (results[0] if results else [])]

    def delete(self, ids: list[int]) -> None:
        if ids:
            self.client.delete(collection_name=self.collection_name, ids=ids)
