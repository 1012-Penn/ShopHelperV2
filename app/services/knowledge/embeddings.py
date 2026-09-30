"""BGE-M3 embeddings over the configured OpenAI-compatible endpoint."""

from __future__ import annotations

import math
from typing import Any


class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "BAAI/bge-m3",
        dimensions: int = 1024,
        client: Any | None = None,
    ):
        if not api_key:
            raise ValueError("embedding API key is required")
        self.model = model
        self.dimensions = dimensions
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
        except Exception:
            # Provider exceptions may include request headers; never surface them.
            raise RuntimeError("embedding request failed") from None

        data = response.data
        if len(data) != len(texts):
            raise ValueError("embedding response count does not match input count")
        ordered: list[list[float] | None] = [None] * len(texts)
        for item in data:
            index = item.index
            if not isinstance(index, int) or not 0 <= index < len(texts) or ordered[index] is not None:
                raise ValueError("embedding response contains invalid item indexes")
            vector = [float(value) for value in item.embedding]
            if len(vector) != self.dimensions:
                raise ValueError(f"BGE-M3 embedding must have {self.dimensions} dimensions")
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("embedding response contains a non-finite value")
            ordered[index] = vector
        if any(vector is None for vector in ordered):
            raise ValueError("embedding response is missing an item")
        return [vector for vector in ordered if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
