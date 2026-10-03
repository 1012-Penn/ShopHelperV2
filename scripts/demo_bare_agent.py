"""Run the framework-free loop with scripted responses or the configured LLM."""
import argparse
import json

from app.services.workflow.bare import run_bare_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if not args.live:
        responses = iter([
            {'content': '', 'tool_calls': [{'name': 'query_order', 'id': 'order', 'args': {'order_id': '1001'}}]},
            {'content': '', 'tool_calls': [{'name': 'query_logistics', 'id': 'logistics', 'args': {'order_id': '1001'}}]},
            {'content': '订单1001已发货，物流运输中（模拟数据）。', 'tool_calls': []},
        ])
        def model(messages, max_tokens):
            return next(responses)
        def tool(name, params, call_id):
            print(json.dumps({'tool': name, 'args': params, 'observation': '模拟数据'}, ensure_ascii=False))
            return '已发货' if name == 'query_order' else '运输中'
    else:
        # Only the adapters use SDKs; the loop above has no framework imports.
        from app.services.workflow.demo_adapter import live_bare_adapters
        model, tool = live_bare_adapters()
    result = run_bare_agent([{'role': 'user', 'content': '请先查订单1001状态，再查它的物流'}], model, tool)
    print(json.dumps({k: v for k, v in result.items() if k != 'messages'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
