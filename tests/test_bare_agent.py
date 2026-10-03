import pytest


def run_loop(*args, **kwargs):
    from app.services.workflow.bare import run_bare_agent
    return run_bare_agent(*args, **kwargs)


def test_order_observation_drives_logistics():
    seen = []
    responses = iter([
        {'content': '', 'tool_calls': [{'id': 'o', 'name': 'query_order', 'args': {'order_id': '1001'}}]},
        {'content': '', 'tool_calls': [{'id': 'l', 'name': 'query_logistics', 'args': {'order_id': '1001'}}]},
        {'content': '物流运输中', 'tool_calls': []},
    ])
    def model(messages, max_tokens):
        seen.append(list(messages))
        return next(responses)
    result = run_loop([{'role': 'user', 'content': '先查订单再查物流'}], model,
                      lambda name, args, call_id: '模拟数据')
    assert result['answer'] == '物流运输中'
    assert result['tool_calls'] == 2
    assert seen[1][-1]['tool_call_id'] == 'o'
    assert seen[2][-1]['tool_call_id'] == 'l'


def test_missing_information_converges_without_tools():
    result = run_loop([{'role': 'user', 'content': '查物流'}],
                      lambda messages, max_tokens: {'content': '请提供订单号', 'tool_calls': []},
                      lambda *args: pytest.fail('must not execute tool'))
    assert result['answer'] == '请提供订单号'
    assert result['decisions'] == 1


def test_repeated_tool_request_stops_and_does_not_execute_again():
    result = run_loop([{'role': 'user', 'content': '订单'}],
                      lambda messages, max_tokens: {'content': '', 'tool_calls': [
                          {'id': str(len(messages)), 'name': 'query_order', 'args': {'order_id': '1001'}}]},
                      lambda *args: '模拟订单')
    assert result['stop_reason'] == 'repeated_tool'
    assert result['tool_calls'] == 1


def test_budget_exhaustion_prevents_model_call():
    from app.services.workflow.policy import Limits
    result = run_loop([{'role': 'user', 'content': '很长问题' * 50}],
                      lambda *args: pytest.fail('budget must stop model'), lambda *args: '',
                      limits=Limits(max_tokens=30))
    assert result['stop_reason'] == 'token_budget'
    assert result['decisions'] == 0


def test_tool_execution_limit_pairs_every_advertised_call():
    from app.services.workflow.policy import Limits
    calls = [{'id': str(i), 'name': 'query_order', 'args': {'order_id': str(i)}} for i in range(3)]
    result = run_loop([{'role': 'user', 'content': '查订单'}],
                      lambda messages, max_tokens: {'content': '', 'tool_calls': calls},
                      lambda *args: 'order', limits=Limits(max_tool_calls=1))
    assert result['tool_calls'] == 1
    assert result['stop_reason'] == 'tool_limit'
    assert [m['tool_call_id'] for m in result['messages'] if m['role'] == 'tool'] == ['0', '1', '2']


def test_invalid_limits_rejected():
    from app.services.workflow.policy import Limits
    with pytest.raises(ValueError):
        Limits(max_decisions=0)
