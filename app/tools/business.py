"""Business tools used by customer support chat."""

from __future__ import annotations

import json
import secrets
from random import choice, randint

from langchain.tools import tool

from app.db.models import Ticket
from app.services.workflow.demo_orders import get_demo_order


def build_tools(session_factory, conversation_id: str, faq_retriever=None):
    @tool
    def query_order(order_id: str) -> str:
        """查询订单的演示数据。输入订单号。"""
        result = get_demo_order(order_id, "demo-user") or {
            "source": "模拟数据",
            "order_id": order_id,
            "found": False,
            "message": "未找到演示订单",
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def query_product(product_query: str) -> str:
        """查询商品的演示数据。输入商品关键词或编号。"""
        result = {
            "source": "模拟数据",
            "product_query": product_query,
            "name": f"{product_query} 精选款",
            "price": round(randint(990, 49900) / 100, 2),
            "in_stock": choice([True, False]),
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def query_faq(query: str) -> str:
        """查询电商商品、配送、退换货、支付或售后知识。输入用户问题。"""
        matches = faq_retriever.search(query)[:5] if faq_retriever is not None else []
        if not matches:
            return json.dumps({"matched": False, "message": f"FAQ 未命中关键词：{query}"}, ensure_ascii=False)
        return json.dumps(
            {
                "matched": True,
                "items": [
                    {"question": hit.question, "answer": hit.answer, "category": hit.category}
                    for hit in matches
                ],
            },
            ensure_ascii=False,
        )

    @tool
    def create_ticket(description: str, ticket_type: str) -> str:
        """为当前会话创建人工工单。输入问题描述和工单类型。"""
        ticket_no = f"TKT-{secrets.token_hex(4).upper()}"
        with session_factory.begin() as session:
            session.add(
                Ticket(
                    ticket_no=ticket_no,
                    conversation_id=conversation_id,
                    description=description,
                    ticket_type=ticket_type,
                    status="open",
                )
            )
        return json.dumps({"ticket_no": ticket_no, "status": "open"}, ensure_ascii=False)

    return [query_order, query_product, query_faq, create_ticket]


def build_definitions(session_factory, faq_retriever=None):
    """Register metadata once; bind the current conversation at execution time."""
    from app.tools.definitions import ToolDefinition
    definitions=[]
    for builtin in build_tools(session_factory, '', faq_retriever):
        schema=builtin.get_input_schema().model_json_schema()
        schema['additionalProperties']=False
        for prop in schema.get('properties',{}).values():
            if prop.get('type')=='string':
                prop.update(minLength=1, pattern=r'\S')
        def handler(args, context, name=builtin.name):
            current={t.name:t for t in build_tools(session_factory, context.conversation_id,faq_retriever)}
            return current[name].invoke(args)
        definitions.append(ToolDefinition(builtin.name,builtin.description,schema,'builtin',handler))
    return definitions
