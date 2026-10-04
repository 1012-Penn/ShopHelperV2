"""Create the additive ch06 demo refund table."""

import argparse
import sys
from pathlib import Path

from sqlalchemy.schema import CreateIndex, CreateTable

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    # Direct `python scripts/migrate_ch06.py` execution puts only `scripts/`
    # first on sys.path; prefer this checkout over another editable install.
    sys.path.insert(0, str(REPOSITORY_ROOT))


def migrate_ch06(engine, *, dry_run=False):
    from app.db.models import RefundApplication

    table = RefundApplication.__table__
    statements = [str(CreateTable(table).compile(engine))]
    statements.extend(
        str(CreateIndex(index).compile(engine)) for index in sorted(table.indexes, key=lambda item: item.name)
    )
    statement = ";\n\n".join(statements)
    if dry_run:
        return statement
    RefundApplication.__table__.create(engine, checkfirst=True)
    return "refund_applications table is present"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the additive table DDL without applying it"
    )
    args = parser.parse_args()
    from app.config import Settings
    from app.db.session import make_engine

    engine = make_engine(Settings.database_url_from_env())
    try:
        print(migrate_ch06(engine, dry_run=args.dry_run))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
