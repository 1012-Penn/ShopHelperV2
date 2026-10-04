"""Mock logistics MCP server. Run with ``python -m mcp_servers.logistics``."""

from __future__ import annotations

import os
import random

from mcp.server.fastmcp import FastMCP

server = FastMCP(
    "MewHelp Logistics",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", os.getenv("LOGISTICS_MCP_PORT", "8765"))),
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
)


@server.tool()
def query_logistics(order_id: str) -> dict:
    """查询订单物流轨迹，返回承运商、当前状态和模拟轨迹。"""
    if order_id == "IS_ERROR":
        raise ValueError("物流查询失败：模拟的 MCP 工具错误")
    steps = ["已揽收", "运输中", "派送中", "已签收"]
    count = random.randint(1, len(steps))
    selected = steps[:count]
    return {
        "order_id": order_id,
        "carrier": random.choice(["顺丰速运", "中通快递", "圆通速递"]),
        "current_status": selected[-1],
        "trajectory": selected,
        "mock_data": True,
        "notice": "模拟数据",
    }


if os.getenv("LOGISTICS_EXTRA_TOOL") == "query_eta":

    @server.tool()
    def query_eta(order_id: str) -> dict:
        """查询模拟的物流预计送达时间。"""
        return {
            "order_id": order_id,
            "estimated_delivery": "明天 18:00 前",
            "mock_data": True,
            "notice": "模拟数据",
        }


if __name__ == "__main__":
    server.run(transport="streamable-http")
