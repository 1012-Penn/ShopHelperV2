"""Deterministic HTTP fixture backed by the real workflow and temporary SQLite files."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(REPOSITORY_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage


class BrowserScriptModel:
    """Small deterministic model boundary; graph, tools, checkpoints and DB stay real."""

    def bind(self, **_kwargs):
        return self

    def bind_tools(self, _tools, **_kwargs):
        return self

    def invoke(self, messages):
        system = str(messages[0].content)
        latest = "\n".join(str(message.content) for message in messages[1:])
        if "本轮问题独立化节点" in system:
            question = latest.split("本轮原问题：\n", 1)[-1]
            return AIMessage(content=json.dumps({"question": question, "reference_resolved": True}, ensure_ascii=False))
        if "本轮意图选择节点" in system:
            return AIMessage(content='{"intent":"售后","confidence":0.99}')
        if "工具决策节点" in system:
            if any(isinstance(message, ToolMessage) for message in messages):
                return AIMessage(content='{"next":"answer","missing":"","actions":[]}')
            if "耳机无法充电" not in latest:
                return AIMessage(content='{"next":"clarify","missing":"问题描述","actions":[]}')
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "create_ticket",
                    "args": {"description": "耳机无法充电", "ticket_type": "售后"},
                    "id": "browser-ticket-call",
                    "type": "tool_call",
                }],
            )
        return AIMessage(content='{"next":"answer","missing":"","actions":[]}')

    def stream(self, messages):
        tool_text = "\n".join(
            str(message.content) for message in messages if isinstance(message, ToolMessage)
        )
        if '"ticket_no"' in tool_text:
            result = json.loads(tool_text[tool_text.find("{"):tool_text.rfind("}") + 1])
            text = f"工单已创建，工单号：{result['ticket_no']}。"
        elif any(isinstance(message, ToolMessage) and message.status == "error" for message in messages):
            text = "已取消本次建单，没有创建工单。"
        else:
            text = "请补充问题描述。"
        yield AIMessageChunk(content=text)


def serve(data_dir: Path, host: str, port: int) -> None:
    import uvicorn
    from workflow_helpers import make_workflow

    from app.main import create_app

    data_dir.mkdir(parents=True, exist_ok=True)
    service = make_workflow(data_dir, model=BrowserScriptModel())
    application = create_app(chat_service=service)
    print(json.dumps({"ready": True, "db": str(data_dir / "business.db")}), flush=True)
    uvicorn.Server(uvicorn.Config(application, host=host, port=port, log_level="warning")).run()


def inspect(db_path: Path, conversation_id: str) -> None:
    connection = sqlite3.connect(db_path)
    try:
        tickets = connection.execute(
            "SELECT ticket_no, description, ticket_type, status FROM tickets WHERE conversation_id=?",
            (conversation_id,),
        ).fetchall()
        audits = connection.execute(
            "SELECT tool_name, status FROM tool_audit_logs WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        print(json.dumps({"tickets": tickets, "audits": audits}, ensure_ascii=False))
    finally:
        connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--inspect", type=Path)
    parser.add_argument("--conversation-id")
    args = parser.parse_args()
    if args.inspect:
        if not args.conversation_id:
            parser.error("--inspect requires --conversation-id")
        inspect(args.inspect, args.conversation_id)
    elif args.data_dir is not None and args.port is not None:
        serve(args.data_dir, args.host, args.port)
    else:
        parser.error("provide --inspect or both --data-dir and --port")
