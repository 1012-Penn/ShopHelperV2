"""Inspect and disposition persistent faithfulness cases."""
import argparse
from app.config import Settings
from app.db.session import make_engine,make_session_factory,create_tables
from app.services.quality.ledger import QualityLedger


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-id');p.add_argument('--status',choices=['未解决','已解决','无需解决'])
    p.add_argument('--resolution',default='')
    args=p.parse_args(argv)
    engine=make_engine(Settings.database_url_from_env());create_tables(engine)
    try:
        ledger=QualityLedger(make_session_factory(engine))
        if args.eval_id:
            ledger.resolve(args.eval_id,args.status,args.resolution)
        for c in ledger.list_cases(None if args.eval_id else args.status):
            print(c.eval_id,c.status,c.seen_count,c.reason,c.resolution or '')
        return 0
    finally:engine.dispose()


if __name__=='__main__':raise SystemExit(main())
