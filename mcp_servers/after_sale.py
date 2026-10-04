"""Mock warranty and return MCP server. Run with ``python -m mcp_servers.after_sale``."""

from __future__ import annotations

import os
import random

from mcp.server.fastmcp import FastMCP

server = FastMCP(
    "MewHelp After Sale",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", os.getenv("AFTER_SALE_MCP_PORT", "8766"))),
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
)


@server.tool()
def query_warranty(order_id: str) -> dict:
    """查询订单商品的模拟保修状态和期限。"""
    under_warranty = random.choice([True, False])
    return {
        "order_id": order_id,
        "under_warranty": under_warranty,
        "warranty_until": "2027-10-04" if under_warranty else None,
        "mock_data": True,
        "notice": "模拟数据",
    }


@server.tool()
def query_return_progress(order_id: str) -> dict:
    """查询订单退货申请的模拟处理进度。"""
    return {
        "order_id": order_id,
        "return_status": random.choice(["审核中", "等待寄回", "退款处理中", "已完成"]),
        "mock_data": True,
        "notice": "模拟数据",
    }


if __name__ == "__main__":
    server.run(transport="streamable-http")
