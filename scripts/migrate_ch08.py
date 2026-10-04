"""Add independent tool audit, write receipt and ticket intent tables."""

from app.config import Settings
from app.db.models import TicketIntent, ToolAuditLog, ToolWriteReceipt
from app.db.session import make_engine


def migrate_ch08(engine):
    for model in (ToolAuditLog, ToolWriteReceipt, TicketIntent):
        model.__table__.create(engine, checkfirst=True)


if __name__ == "__main__":
    engine = make_engine(Settings.database_url_from_env())
    try:
        migrate_ch08(engine)
        print("ch08 tables ready")
    finally:
        engine.dispose()
