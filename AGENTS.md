# MewHelp — AI 协作规范（AGENTS）

本项目**声明使用 superpowers 技能套件**（插件 `superpowers@claude-plugins-official`）。
在本项目中工作的任何 AI 助手必须遵循以下 superpowers 工作流：

## 必用工作流

| 场景 | 必须使用的技能 |
|---|---|
| 需求不明确、需要设计 | `superpowers:brainstorming` → `superpowers:writing-plans` |
| 动手实现 | `superpowers:executing-plans`（先有计划，后写代码） |
| 编写/修改代码 | `superpowers:test-driven-development`（红-绿-重构） |
| 遇到 bug | `superpowers:systematic-debugging`（先定位根因，禁止猜测式乱改） |
| 并行子任务 | `superpowers:dispatching-parallel-agents` |
| 完成前验证 | `superpowers:verification-before-completion` |
| 合并/提交前 | `superpowers:requesting-code-review`；收到评审用 `superpowers:receiving-code-review` |
| 分支收尾 | `superpowers:finishing-a-development-branch` |

## 硬性要求

- 未做头脑风暴/计划前，不得直接开始大功能实现
- 未写测试前，不得提交新功能代码
- 声称"完成"前，必须先执行验证技能
