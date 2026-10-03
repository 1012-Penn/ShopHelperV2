"""Build the versioned native BM25 collection without deleting the old index."""
import argparse
import os
from dotenv import dotenv_values
from app.config import Settings
from app.db.session import make_engine, make_session_factory, create_tables
from app.services.knowledge.embeddings import EmbeddingClient
from app.services.knowledge.hybrid_store import HybridStore
from app.services.quality.index import HybridIndexer


def build_indexer():
    settings=Settings.from_env(require_chat=False)
    values={**dotenv_values('.env'),**os.environ}
    engine=make_engine(settings.database_url)
    create_tables(engine)
    embeddings=EmbeddingClient(api_key=settings.embedding_api_key,base_url=settings.embedding_api_base,model=settings.embedding_model)
    store=HybridStore(uri=settings.milvus_uri or 'http://localhost:19530',collection_name=values.get('HYBRID_COLLECTION','knowledge_ch04'))
    return HybridIndexer(make_session_factory(engine),embeddings,store)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args(argv)
    indexer=None
    try:
        indexer=build_indexer()
        print(indexer.sync(args.batch_size))
        return 0
    except Exception:
        print('Hybrid rebuild failed; check configuration and service availability; safe to retry.')
        return 1
    finally:
        if indexer:
            indexer.store.close()
            indexer.embeddings.close()
            indexer.session_factory.kw['bind'].dispose()


if __name__=='__main__':
    raise SystemExit(main())
