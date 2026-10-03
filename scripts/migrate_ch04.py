"""Explicit, repeatable schema upgrade; never rebuild conversation tables."""

from sqlalchemy import inspect, text


def _upgrade_mysql_identity(connection, column):
    """Avoid redundant ALTERs and restore incoming keys for the legacy nullable ID."""
    if (
        column.get("autoincrement")
        and not column.get("nullable")
        and getattr(column["type"], "unsigned", False)
    ):
        return
    inspector = inspect(connection)
    quote = connection.dialect.identifier_preparer.quote
    schema = connection.execute(text("SELECT DATABASE()")).scalar()
    keys = []
    for table in inspector.get_table_names():
        for key in inspector.get_foreign_keys(table):
            if (
                key["referred_table"] == "conversations"
                and "id" in key["referred_columns"]
                and key.get("referred_schema") in (None, schema)
            ):
                columns = ",".join(quote(name) for name in key["constrained_columns"])
                referred = ",".join(quote(name) for name in key["referred_columns"])
                restore = (
                    f"ALTER TABLE {quote(table)} ADD CONSTRAINT {quote(key['name'])} "
                    f"FOREIGN KEY ({columns}) REFERENCES conversations ({referred})"
                )
                for option, sql_name in (
                    ("ondelete", "ON DELETE"),
                    ("onupdate", "ON UPDATE"),
                ):
                    action = key.get("options", {}).get(option)
                    if action:
                        if action not in {
                            "CASCADE",
                            "RESTRICT",
                            "SET NULL",
                            "NO ACTION",
                            "SET DEFAULT",
                        }:
                            raise ValueError("unsupported foreign key action")
                        restore += f" {sql_name} {action}"
                keys.append((table, key["name"], restore))
    dropped = []
    try:
        for table, name, restore in keys:
            connection.execute(
                text(f"ALTER TABLE {quote(table)} DROP FOREIGN KEY {quote(name)}")
            )
            dropped.append(restore)
        connection.execute(
            text(
                "ALTER TABLE conversations MODIFY COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT"
            )
        )
    finally:
        for restore in dropped:
            connection.execute(text(restore))


def migrate_ch04(engine):
    with engine.connect() as c:
        mysql = engine.dialect.name == "mysql"
        if (
            mysql
            and c.execute(text("SELECT GET_LOCK('mewhelp_ch04_schema', 15)")).scalar()
            != 1
        ):
            raise RuntimeError("schema migration lock unavailable")
        try:
            tables = set(inspect(c).get_table_names())
            if "conversations" in tables:
                columns = {
                    col["name"]: col for col in inspect(c).get_columns("conversations")
                }
                if "id" not in columns:
                    if mysql:
                        c.execute(
                            text(
                                "ALTER TABLE conversations ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, ADD UNIQUE KEY uk_conversations_id (id)"
                            )
                        )
                    else:
                        c.execute(
                            text("ALTER TABLE conversations ADD COLUMN id INTEGER")
                        )
                if mysql:
                    if "id" in columns:
                        _upgrade_mysql_identity(c, columns["id"])
                else:
                    c.execute(
                        text("UPDATE conversations SET id=rowid WHERE id IS NULL")
                    )
                    c.execute(
                        text(
                            "CREATE UNIQUE INDEX IF NOT EXISTS uk_conversations_id ON conversations(id)"
                        )
                    )
                    c.execute(
                        text(
                            "CREATE TABLE IF NOT EXISTS conversation_identity_sequence (id INTEGER PRIMARY KEY AUTOINCREMENT)"
                        )
                    )
                    c.execute(
                        text(
                            "INSERT OR IGNORE INTO conversation_identity_sequence(id) SELECT MAX(id) FROM conversations WHERE id IS NOT NULL"
                        )
                    )
                    c.execute(
                        text("""CREATE TRIGGER IF NOT EXISTS conversations_numeric_id AFTER INSERT ON conversations
                        WHEN NEW.id IS NULL BEGIN
                        INSERT INTO conversation_identity_sequence(id) VALUES (NULL);
                        UPDATE conversations SET id=last_insert_rowid() WHERE conversation_id=NEW.conversation_id;
                        END""")
                    )
            if "messages" in tables and "citations" not in {
                col["name"] for col in inspect(c).get_columns("messages")
            }:
                c.execute(text("ALTER TABLE messages ADD COLUMN citations JSON NULL"))
            if mysql and "faith_cases" in tables:
                c.execute(
                    text(
                        "ALTER TABLE faith_cases MODIFY strategy VARCHAR(24) NOT NULL DEFAULT 'hybrid_rerank', MODIFY status ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决', MODIFY seen_count INT UNSIGNED NOT NULL DEFAULT 1"
                    )
                )
            c.commit()
        finally:
            if mysql:
                c.execute(text("SELECT RELEASE_LOCK('mewhelp_ch04_schema')"))
                c.commit()


def main():
    from app.config import Settings
    from app.db.session import create_tables, make_engine

    engine = make_engine(Settings.database_url_from_env())
    try:
        create_tables(engine)
        print("ch04 migration complete")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
