"""Single-turn normalization preserving identifiers and negative conditions."""
import re
from functools import lru_cache
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(frozen=True)
class QueryUnderstanding:
    raw: str
    canonical: str
    lexical: str


class NormalizedQuery(BaseModel):
    canonical: str


class QueryNormalizer:
    def __init__(self, rewrite=None):
        self.rewrite=rewrite

    @lru_cache(maxsize=1024)
    def normalize(self, raw):
        canonical=raw.strip()
        if self.rewrite:
            try:
                candidate=self.rewrite(raw)
                identifiers=re.findall(r'[A-Za-z0-9][A-Za-z0-9_.-]*',raw)
                candidate_identifiers=re.findall(r'[A-Za-z0-9][A-Za-z0-9_.-]*',candidate)
                if candidate.strip() and set(identifiers)==set(candidate_identifiers) and all(token not in raw or token in candidate for token in ['不','无','未','没']):
                    canonical=candidate.strip()
            except Exception:
                pass  # malformed normalization falls back to original question
        expansions=[]
        for words in [('邮费','运费'),('退钱','退款'),('退东西','退货'),('不开机','无法开机'),('多久到','配送时效')]:
            if any(word in raw or word in canonical for word in words):
                expansions.extend(words)
        return QueryUnderstanding(raw,canonical,' '.join(dict.fromkeys([raw,canonical,*expansions])))

    @classmethod
    def from_model(cls, model):
        structured=model.with_structured_output(NormalizedQuery,method='json_mode')
        def rewrite(raw):
            result=structured.invoke([{'role':'system','content':'将本轮口语客服问题归一为标准问法，只看本轮。不添加型号、数字、品类，不丢否定条件。输出JSON：{"canonical":"标准问法"}。'}, {'role':'user','content':raw}])
            return result.canonical
        return cls(rewrite)
