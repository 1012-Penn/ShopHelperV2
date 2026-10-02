"""Explicit, repeatable schema upgrade; never rebuild conversation tables."""
from sqlalchemy import inspect, text


def migrate_ch04(engine):
    with engine.connect() as c:
        mysql = engine.dialect.name == 'mysql'
        if mysql and c.execute(text("SELECT GET_LOCK('mewhelp_ch04_schema', 15)")).scalar() != 1:
            raise RuntimeError('schema migration lock unavailable')
        try:
            tables = set(inspect(c).get_table_names())
            if 'conversations' in tables:
                columns = {col['name']: col for col in inspect(c).get_columns('conversations')}
                if 'id' not in columns:
                    if mysql:
                        c.execute(text('ALTER TABLE conversations ADD COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, ADD UNIQUE KEY uk_conversations_id (id)'))
                    else:
                        c.execute(text('ALTER TABLE conversations ADD COLUMN id INTEGER'))
                if mysql:
                    c.execute(text('ALTER TABLE conversations MODIFY COLUMN id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT'))
                else:
                    c.execute(text('UPDATE conversations SET id=rowid WHERE id IS NULL'))
                    c.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS uk_conversations_id ON conversations(id)'))
                    c.execute(text('CREATE TABLE IF NOT EXISTS conversation_identity_sequence (id INTEGER PRIMARY KEY AUTOINCREMENT)'))
                    c.execute(text('INSERT OR IGNORE INTO conversation_identity_sequence(id) SELECT MAX(id) FROM conversations WHERE id IS NOT NULL'))
                    c.execute(text('''CREATE TRIGGER IF NOT EXISTS conversations_numeric_id AFTER INSERT ON conversations
                        WHEN NEW.id IS NULL BEGIN
                        INSERT INTO conversation_identity_sequence(id) VALUES (NULL);
                        UPDATE conversations SET id=last_insert_rowid() WHERE conversation_id=NEW.conversation_id;
                        END'''))
            if 'messages' in tables and 'citations' not in {col['name'] for col in inspect(c).get_columns('messages')}:
                c.execute(text('ALTER TABLE messages ADD COLUMN citations JSON NULL'))
            if mysql and 'faith_cases' in tables:
                c.execute(text("ALTER TABLE faith_cases MODIFY strategy VARCHAR(24) NOT NULL DEFAULT 'hybrid_rerank', MODIFY status ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决', MODIFY seen_count INT UNSIGNED NOT NULL DEFAULT 1"))
            c.commit()
        finally:
            if mysql:
                c.execute(text("SELECT RELEASE_LOCK('mewhelp_ch04_schema')"))
                c.commit()


def main():
    from app.config import Settings
    from app.db.session import make_engine, create_tables
    engine = make_engine(Settings.database_url_from_env())
    try:
        create_tables(engine)
        print('ch04 migration complete')
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
