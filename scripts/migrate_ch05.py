"""Repeatable, additive Message action metadata migration."""

from sqlalchemy import inspect, text


def migrate_ch05(engine):
    with engine.connect() as connection:
        mysql = engine.dialect.name == "mysql"
        if (
            mysql
            and connection.execute(
                text("SELECT GET_LOCK('mewhelp_ch05_schema',15)")
            ).scalar()
            != 1
        ):
            raise RuntimeError("schema migration lock unavailable")
        try:
            inspector = inspect(connection)
            if "messages" in inspector.get_table_names():
                columns = {c["name"] for c in inspector.get_columns("messages")}
                if "actions" not in columns:
                    connection.execute(
                        text("ALTER TABLE messages ADD COLUMN actions JSON NULL")
                    )
            connection.commit()
        finally:
            if mysql:
                connection.execute(text("SELECT RELEASE_LOCK('mewhelp_ch05_schema')"))
                connection.commit()


def main():
    from app.config import Settings
    from app.db.session import create_tables, make_engine

    engine = make_engine(Settings.database_url_from_env())
    try:
        create_tables(engine)
        print("ch05 migration complete")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
