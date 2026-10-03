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

退货政策需要结合你的具体订单来判断，通用说明不能替代适用于该订单的具体规则。以下是主要要点：

**退换货资格与条件**
退换货资格、申请期限、商品状态要求、配件和包装要求、费用承担及例外情况，以购买时适用的法律规定、店铺公示政策、商品页面和订单详情为准。商品类别、定制属性、使用状态或活动条件都可能影响处理方式。[1]

**如何申请**
在订单页面选择适用的售后类型并填写真实原因；页面无法操作时可联系在线客服。客服核对资格和订单状态后会说明步骤、寄回要求及收件信息。寄回前请以系统确认的地址为准，并保留物流凭证。[4]

**换货**
换货是否可用取决于商品、库存、订单和店铺规则，可从订单售后入口提交申请。[5]

**费用承担**
退货运费及其他费用根据退货原因、适用政策、订单约定和核实结果确定，需先核对是否为顾客原因、商品质量问题或履约问题。[3]

**退款去向与金额**
退款通常按订单和支付渠道的流程处理，具体去向以订单退款详情及支付渠道记录为准。[2] 部分退款金额根据售后核验、商品明细、优惠分摊、运费和适用规则计算，以系统审核结果为准。[6]

**已使用商品**
能否退货取决于商品类别、使用或拆封情况、质量问题、适用法律和购买时公示的规则，需按具体订单核实。[7]

如果你告诉我具体订单和商品情况，我可以帮你进一步核对适用规则。

Path: refer → classify → route → retrieve → gate → agent → agent_answer → log

## logistics

订单 1001 的物流到哪了

订单 1001 的物流信息如下（模拟数据）：

- 快递公司：中通快递
- 当前状态：派送中
- 最近更新：刚刚

如需更精确的位置或派送员信息，建议联系快递公司核实。

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

- **订单状态**：已签收，订单金额 259.06 元
- **物流轨迹**：承运商顺丰速运，当前状态为「派送中」，最后更新时间为「刚刚」

需要说明的是，订单状态显示已签收，而物流状态显示派送中，两者存在不一致，建议以实际物流信息为准或进一步核实。

Path: refer → classify → route → agent → tools → agent → tools → agent → agent_answer → log
