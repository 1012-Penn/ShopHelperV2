# ch05 graph demonstration (live)

Tools for orders/logistics remain chapter-2 simulated business data. Complaint actions are suggestions only.

- all_completed: True
- policy_forced_retrieval: True
- logistics_agent_tool: False
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

- 承运商：顺丰速运
- 当前状态：运输中
- 最新更新：刚刚

目前只能看到“运输中”，没有更具体的当前位置或中转节点信息。如需更详细的轨迹，建议通过顺丰官方渠道用运单号查询。

Path: refer → classify → route → agent → tools → agent → tools → agent → agent_answer → log

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

订单 1001 的查询结果如下（均为模拟数据）：

- **订单状态**：已签收，订单金额 684.74
- **物流轨迹**：承运商顺丰速运，物流状态已签收，最后更新时间为“刚刚”

如需更详细的物流节点信息，建议通过顺丰速运官方渠道进一步核实。

Path: refer → classify → route → agent → tools → agent → tools → agent → agent_answer → log
