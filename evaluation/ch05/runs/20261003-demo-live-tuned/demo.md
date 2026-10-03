# ch05 graph demonstration (live)

Tools for orders/logistics remain chapter-2 simulated business data. Complaint actions are suggestions only.

- all_completed: True
- policy_forced_retrieval: True
- logistics_agent_tool: True
- complaint_two_suggestions: True
- chitchat_zero_model: True
- complex_multiple_tools: True

## policy

退货政策是什么

抱歉，现有知识库没有足够证据回答这个问题，我无法确认。建议联系人工客服核实。

Path: refer → classify → route → retrieve → gate → fixed → log

## logistics

订单 1001 的物流到哪了

订单 1001 的物流信息如下（模拟数据）：

- 承运商：圆通速递
- 当前状态：已签收
- 最后更新：刚刚

如需了解签收人、签收时间或具体派送网点等更详细信息，可以告诉我，我再帮你查。

Path: refer → classify → route → agent → tools → agent → agent_answer → log

## complaint

我要投诉

很抱歉给您带来不好的体验，我们重视您的反馈。您可以选择转人工客服或创建工单跟进。

Path: refer → classify → route → fixed → log

## chitchat

你好

您好，我是客服小猫，可以帮您查询订单、物流、商品信息和售后问题。

Path: refer → classify → route → fixed → log

## complex

先查订单1001状态，再查询它的物流轨迹

本轮查询已达到处理上限，请补充信息或联系人工客服核实。

Path: refer → classify → route → agent → tools → agent → tools → agent → agent_answer → log
