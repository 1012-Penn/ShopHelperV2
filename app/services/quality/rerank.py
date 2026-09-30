"""Strict adapter for the fixed bge-reranker-v2-m3 model."""
import math
import httpx


class Reranker:
    model='BAAI/bge-reranker-v2-m3'
    def __init__(self, api_key, base_url='https://api.siliconflow.cn/v1', client=None):
        if not api_key:
            raise ValueError('RERANK_API_KEY required')
        self.api_key=api_key
        self.base_url=base_url.rstrip('/')
        self.client=client or httpx.Client(timeout=45)

    def rank(self, query, documents, top_n=10):
        if not documents:
            return []
        try:
            response=self.client.post(self.base_url+'/rerank',headers={'Authorization':'Bearer '+self.api_key},json={'model':self.model,'query':query,'documents':documents,'top_n':min(top_n,len(documents)),'return_documents':False})
            response.raise_for_status()
        except httpx.HTTPError:
            raise RuntimeError('rerank request failed') from None
        results=response.json()['results']
        seen=set();ranked=[]
        for result in results:
            index=result['index'];score=result['relevance_score']
            if type(index) is not int or not 0<=index<len(documents) or index in seen or not isinstance(score,(int,float)) or not math.isfinite(score):
                raise ValueError('invalid rerank result')
            seen.add(index);ranked.append((index,float(score)))
        if len(ranked)!=min(top_n,len(documents)):
            raise ValueError('incomplete rerank result')
        return sorted(ranked,key=lambda r:r[1],reverse=True)

    def close(self):
        self.client.close()
