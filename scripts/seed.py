"""Create the schema and insert the sample FAQ rows."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import FAQ
from app.db.session import create_tables, make_engine, make_session_factory


SAMPLE_FAQ = (
    ("退货政策是什么？", "商品签收后 7 天内可申请退货，商品需保持完好。", "售后政策"),
    ("配送需要多久？", "一般订单会在 1 至 3 个工作日内送达。", "配送"),
    ("支持哪些支付方式？", "目前支持银行卡、微信支付和支付宝。", "支付"),
    ("退款多久到账？", "审核通过后，退款通常会在 3 至 5 个工作日内退回原支付账户。", "退款"),
)


def seed_faq(session: Session) -> None:
    existing = set(session.scalars(select(FAQ.question)))
    for question, answer, category in SAMPLE_FAQ:
        if question not in existing:
            session.add(FAQ(question=question, answer=answer, category=category))
    session.flush()


def main() -> None:
    engine = make_engine(Settings.database_url_from_env())
    create_tables(engine)
    session_factory = make_session_factory(engine)
    with session_factory.begin() as session:
        seed_faq(session)
    print("FAQ seed complete")


if __name__ == "__main__":
    main()
