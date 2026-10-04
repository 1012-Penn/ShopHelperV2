"""Milvus-native BM25 and dense retrieval with identical scalar prefilters."""
import json
import math
from dataclasses import dataclass
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker


@dataclass(frozen=True)
class HybridRow:
    chunk_id: int
    vector: list[float]
    text: str
    category: str
    content_hash: str


@dataclass(frozen=True)
class HybridHit:
    chunk_id: int
    score: float
    content_hash: str


class HybridStore:
    def __init__(self, uri='http://localhost:19530', collection_name='knowledge_ch04', client=None):
        self.collection_name=collection_name
        self.client=client if client is not None else MilvusClient(uri=uri)

    def ensure_collection(self):
        if not self.client.has_collection(self.collection_name):
            schema=self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field('chunk_id',DataType.INT64,is_primary=True)
            schema.add_field('embedding',DataType.FLOAT_VECTOR,dim=1024)
            schema.add_field('text',DataType.VARCHAR,max_length=65535,enable_analyzer=True,analyzer_params={'type':'chinese'})
            schema.add_field('bm25',DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field('category',DataType.VARCHAR,max_length=512)
            schema.add_field('is_active',DataType.BOOL)
            schema.add_field('content_hash',DataType.VARCHAR,max_length=64)
            schema.add_function(Function(name='text_bm25',input_field_names=['text'],output_field_names=['bm25'],function_type=FunctionType.BM25))
            indexes=self.client.prepare_index_params()
            indexes.add_index(field_name='embedding',index_type='AUTOINDEX',metric_type='COSINE')
            indexes.add_index(field_name='bm25',index_type='SPARSE_INVERTED_INDEX',metric_type='BM25')
            self.client.create_collection(collection_name=self.collection_name,schema=schema,index_params=indexes,consistency_level='Strong')
        description=self.client.describe_collection(self.collection_name)
        fields={f['name']:f for f in description['fields']}
        if not {'chunk_id','embedding','text','bm25','category','is_active','content_hash'} <= fields.keys():
            raise ValueError('incompatible hybrid collection; choose a new collection name')
        text_params=fields['text'].get('params',{})
        analyzer=text_params.get('analyzer_params',{})
        if isinstance(analyzer,str):
            analyzer=json.loads(analyzer)
        if analyzer.get('type')!='chinese' or int(fields['embedding'].get('params',{}).get('dim',0))!=1024:
            raise ValueError('incompatible analyzer or embedding dimension')
        if not any(f.get('type') == FunctionType.BM25 or f.get('type') == int(FunctionType.BM25) for f in description.get('functions',[])):
            raise ValueError('native BM25 function missing')
        self.client.load_collection(collection_name=self.collection_name)

    def upsert(self, rows):
        if not rows:
            return []
        for r in rows:
            if len(r.vector)!=1024 or not all(math.isfinite(x) for x in r.vector):
                raise ValueError('invalid 1024-dimensional embedding')
            if len(r.text.encode())>65535 or len(r.category.encode())>512:
                raise ValueError('hybrid field exceeds Milvus byte limit')
        data=[dict(chunk_id=r.chunk_id,embedding=r.vector,text=r.text,category=r.category,content_hash=r.content_hash,is_active=True) for r in rows]
        ids=self.client.upsert(collection_name=self.collection_name,data=data)['ids']
        if ids!=[r.chunk_id for r in rows]:
            raise RuntimeError('hybrid upsert ids mismatch')
        return ids

    def search(self, query_vector, lexical_query, strategy, category=None, limit=50,
               category_prefixes=None, chunk_ids=None):
        if strategy not in {'dense','bm25','hybrid','hybrid_rerank'}:
            raise ValueError('unknown strategy')
        expr='is_active == true'
        if category is not None:
            if not category.strip() or len(category.encode())>512:
                raise ValueError('invalid category filter')
            expr+=' and category == '+json.dumps(category,ensure_ascii=False)
        if category_prefixes:
            if any(not p.strip() or len(p.encode()) > 512 for p in category_prefixes):
                raise ValueError('invalid category prefix filter')
            clauses = [
                'category like ' + json.dumps(prefix + '%', ensure_ascii=False)
                for prefix in category_prefixes
            ]
            expr += ' and (' + ' or '.join(clauses) + ')'
        if chunk_ids is not None:
            ids = tuple(dict.fromkeys(chunk_ids))
            if any(type(chunk_id) is not int or chunk_id < 1 for chunk_id in ids):
                raise ValueError('invalid chunk id filter')
            if not ids:
                return []
            expr += ' and chunk_id in [' + ', '.join(str(chunk_id) for chunk_id in ids) + ']'
        if strategy!='bm25' and (query_vector is None or len(query_vector)!=1024):
            raise ValueError('dense query dimension must be 1024')
        if strategy in {'hybrid','hybrid_rerank'}:
            requests=[AnnSearchRequest(data=[query_vector],anns_field='embedding',param={'metric_type':'COSINE'},limit=50,expr=expr),
                      AnnSearchRequest(data=[lexical_query],anns_field='bm25',param={'metric_type':'BM25'},limit=50,expr=expr)]
            result=self.client.hybrid_search(collection_name=self.collection_name,reqs=requests,ranker=RRFRanker(),limit=limit,output_fields=['content_hash'],consistency_level='Strong')
        else:
            result=self.client.search(collection_name=self.collection_name,data=[lexical_query if strategy=='bm25' else query_vector],anns_field='bm25' if strategy=='bm25' else 'embedding',search_params={'metric_type':'BM25' if strategy=='bm25' else 'COSINE'},filter=expr,limit=limit,output_fields=['content_hash'],consistency_level='Strong')
        return [HybridHit(int(h.get('chunk_id',h.get('id'))),float(h['distance']),h['entity']['content_hash']) for h in result[0]] if result else []

    def delete(self, ids):
        if ids:
            self.client.delete(collection_name=self.collection_name,ids=ids)

    def close(self):
        self.client.close()
