"""Create the schema and insert the sample FAQ rows."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import FAQ
from app.db.session import create_tables, make_engine, make_session_factory


SAMPLE_FAQ = (
    (
        "退货政策是什么？",
        "退换货资格、申请期限、商品状态要求及费用承担，以购买时适用的法律规定、店铺公示政策和订单详情为准。",
        "退换货",
    ),
    (
        "配送需要多久？",
        "预计发货和送达安排，以商品页、订单详情及结算页展示的实际履约信息为准；页面信息不明确时请联系在线客服核实。",
        "配送",
    ),
    (
        "支持哪些支付方式？",
        "可使用的支付方式以结算页当前提供的选项为准；请勿向客服提供支付密码或验证码。",
        "支付",
    ),
    (
        "退款多久到账？",
        "退款进度可在订单售后详情查看；到账时间受订单处理和支付渠道影响，具体状态以订单退款记录和支付渠道信息为准。",
        "退款",
    ),
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
