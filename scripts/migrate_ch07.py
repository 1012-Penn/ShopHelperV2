"""Repeatable additive anchors; Base.metadata creates the separate summary table."""

from sqlalchemy import inspect, text


def migrate_ch07(engine):
    with engine.connect() as conn:
        mysql = engine.dialect.name == "mysql"
        if (
            mysql
            and conn.execute(text("SELECT GET_LOCK('mewhelp_ch07_schema',15)")).scalar()
            != 1
        ):
            raise RuntimeError("schema migration lock unavailable")
        try:
            inspector = inspect(conn)
            if "conversations" in inspector.get_table_names():
                columns = {c["name"] for c in inspector.get_columns("conversations")}
                for name in ("summary_upto_msg_id", "layer1_from_msg_id"):
                    if name not in columns:
                        conn.execute(
                            text(
                                f"ALTER TABLE conversations ADD COLUMN {name} INTEGER NULL"
                            )
                        )
            conn.commit()
        finally:
            if mysql:
                conn.execute(text("SELECT RELEASE_LOCK('mewhelp_ch07_schema')"))
                conn.commit()


def main():
    from app.config import Settings
    from app.db.session import create_tables, make_engine

    engine = make_engine(Settings.database_url_from_env())
    try:
        create_tables(engine)
        print("ch07 migration complete")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
