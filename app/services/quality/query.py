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
    downgrade_reason: str = ""


class NormalizedQuery(BaseModel):
    canonical: str


class QueryNormalizer:
    def __init__(self, rewrite=None):
        self.rewrite=rewrite

    @lru_cache(maxsize=1024)
    def normalize(self, raw):
        canonical=raw.strip()
        reason=""
        if self.rewrite:
            try:
                candidate=self.rewrite(raw)
                identifiers=re.findall(r'[A-Za-z0-9][A-Za-z0-9_.-]*',raw)
                candidate_identifiers=re.findall(r'[A-Za-z0-9][A-Za-z0-9_.-]*',candidate)
                chinese_numbers=re.findall(r'[零〇一二三四五六七八九十百千万亿两]+',raw)
                candidate_numbers=re.findall(r'[零〇一二三四五六七八九十百千万亿两]+',candidate)
                negative=bool(re.search(r'[不无未没非别否]',raw))
                if candidate.strip() and sorted(identifiers)==sorted(candidate_identifiers) and chinese_numbers==candidate_numbers and (not negative or candidate.strip()==raw.strip()):
                    canonical=candidate.strip()
                else:
                    reason='改写未完整保留标识、数值或否定范围，使用原话'
            except Exception:
                reason='改写服务或输出不可用，使用原话'
        expansions=[]
        for words in [('邮费','运费'),('退钱','退款'),('退东西','退货'),('不开机','无法开机'),('多久到','配送时效')]:
            if any(word in raw or word in canonical for word in words):
                expansions.extend(words)
        return QueryUnderstanding(raw,canonical,' '.join(dict.fromkeys([raw,canonical,*expansions])),reason)

    @classmethod
    def from_model(cls, model):
        structured=model.with_structured_output(NormalizedQuery,method='json_mode')
        def rewrite(raw):
            result=structured.invoke([{'role':'system','content':'将本轮口语客服问题归一为标准问法，只看本轮。不添加型号、数字、品类，不丢否定条件。输出JSON：{"canonical":"标准问法"}。'}, {'role':'user','content':raw}])
            return result.canonical
        return cls(rewrite)
