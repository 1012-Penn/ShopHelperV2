"""Existing model and business tools adapted to the plain dictionary loop."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.tools.business import build_tools


def live_bare_adapters():
    settings = Settings.from_env()
    tools = [
        t
        for t in build_tools(None, "bare-demo")
        if t.name in {"query_order", "query_product", "query_logistics"}
    ]
    tool_map = {tool.name: tool for tool in tools}
    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        timeout=45,
        max_retries=0,
    )

    def call_model(messages, max_tokens):
        converted = [
            SystemMessage(
                content="你是客服。按要求查订单后查物流，看到工具结果再决定下一步。"
                "不用提供推理过程，没信息就追问；没有需要调用的工具时给简短答复。工具是模拟数据。"
            )
        ]
        for message in messages:
            if message["role"] == "user":
                converted.append(HumanMessage(content=message["content"]))
            elif message["role"] == "tool":
                converted.append(
                    ToolMessage(
                        content=message["content"], tool_call_id=message["tool_call_id"]
                    )
                )
            else:
                converted.append(
                    AIMessage(
                        content=message["content"],
                        tool_calls=message.get("tool_calls") or [],
                    )
                )
        response = model.bind_tools(tools).bind(max_tokens=max_tokens).invoke(converted)
        return {
            "content": response.content,
            "tool_calls": response.tool_calls,
            "usage": response.usage_metadata,
        }

    def run_tool(name, args, call_id):
        return tool_map[name].invoke(args)

    return call_model, run_tool
