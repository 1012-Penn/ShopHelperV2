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

退货政策需要结合你的具体订单来判断，通用说明不能替代适用于该订单的具体规则。主要涉及以下几点：

**退换货条件**
退换货资格、申请期限、商品状态要求、配件和包装要求、费用承担及例外情况，以购买时适用的法律规定、店铺公示政策、商品页面和订单详情为准。商品类别、定制属性、使用状态或活动条件都可能影响处理方式。[1]

**如何申请**
在订单页面选择适用的售后类型并填写真实原因；页面无法操作时可联系在线客服。寄回前请以系统确认的地址和说明为准，并保留物流凭证。[4]

**退货运费**
运费及其他费用根据退货原因、适用政策、订单约定和核实结果确定，需要先核对是否为顾客原因、商品质量或履约问题。[3]

**退款去向**
退款通常按订单和支付渠道的流程处理，具体去向以订单退款详情及支付渠道记录为准。[2]

**其他情况**
- 商品已使用能否退，取决于商品类别、拆封情况、质量问题及公示规则，需按具体订单核实。[7]
- 换货是否可用取决于商品、库存、订单和店铺规则。[5]
- 退回商品可能需要验收，结果以实际核查为准。[9]

如果你告诉我具体订单和商品情况，我可以帮你进一步核对适用规则。

Path: refer → classify → route → retrieve → gate → agent → agent_answer → log

## logistics

订单 1001 的物流到哪了

订单 1001 的物流信息如下（模拟数据）：

- 承运商：顺丰速运
- 当前状态：派送中
- 最近更新：刚刚

如需更精确的位置或派送员信息，建议直接联系承运商核实。

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

订单1001的查询结果如下（均为模拟数据）：

- **订单状态**：待发货，订单金额 537.16 元
- **物流轨迹**：承运商为顺丰速运，当前状态为“运输中”，最后更新时间为“刚刚”

说明：订单状态显示“待发货”，而物流显示“运输中”，两者存在不一致，建议以实际页面信息为准或进一步核实。

Path: refer → classify → route → agent → tools → agent → tools → agent → agent_answer → log
