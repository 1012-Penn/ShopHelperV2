# ch05 graph demonstration (live)

Tools for orders/logistics remain chapter-2 simulated business data. Complaint actions are suggestions only.

- agent_answers_completed: True
- all_completed: True
- policy_forced_retrieval: True
- logistics_agent_tool: True
- complaint_two_suggestions: True
- chitchat_zero_model: True
- complex_multiple_tools: True

## policy

退货政策是什么

退货政策不是统一的固定规则，具体以你购买时适用的法律规定、店铺公示政策、商品页面和订单详情为准。商品类别、是否定制、使用状态、活动条件等都可能影响能否退换及处理方式。[1]

一般流程和注意点：
- **申请入口**：在订单页面选择适用的售后类型并填写真实原因；页面无法操作时可联系在线客服。[4]
- **退货资格**：需先核对订单、商品状态和申请原因，不能一概而论。已使用或拆封的商品能否退，取决于商品类别、质量问题及公示规则。[1][7]
- **运费承担**：根据退货原因、适用政策、订单约定和核实结果确定，未核对前无法承诺由哪一方承担。[3]
- **退款去向**：通常按订单和支付渠道的退款流程处理，具体以订单退款详情及支付渠道记录为准。[2]
- **退回验收**：退回商品可能需要验收，结果以售后规则和实际核查为准。[9]

如果你告诉我具体商品和退货原因，我可以帮你进一步核对适用规则。

Path: refer → classify → route → retrieve → gate → agent → agent_answer → log

## logistics

订单 1001 的物流到哪了

订单 1001 的物流信息如下（模拟数据）：

- 快递公司：中通快递
- 当前状态：已签收
- 最后更新：刚刚

如需查看具体签收地点或签收人，建议您再提供一下收件人信息或直接在快递官网核实。

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

订单 1001 的查询结果如下（均为模拟数据）：

- **订单状态**：待发货，订单金额 251.34
- **物流轨迹**：承运商顺丰速运，当前状态为“派送中”，最后更新时间为“刚刚”

需要说明的是，订单状态显示“待发货”，而物流显示“派送中”，两者存在不一致，建议您以实际物流信息为准或进一步核实。

Path: refer → classify → route → agent → tools → agent → tools → agent → agent_answer → log
