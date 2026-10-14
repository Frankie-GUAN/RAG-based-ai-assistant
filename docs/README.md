# 项目文档索引

> 本项目定位：**个人使用的知识库问答助手**（直播主播经营 Agent 平台已另立项目，其 PRD 已从本仓库移除）。

## 现有文档

| 文档 | 状态 | 说明 |
|---|---|---|
| [技术事实与缺陷清单](tech-notes/verified-facts.md) | **维护中 · 优先读** | 本机实测过的技术前提（pyarrow/torch 段错误、MCP 2.2.0 的类名与字段、pickle 兼容性）+ 带行号的已确认缺陷 + 已知未修项。**改代码前必读** |
| [修改计划：个人知识库助手](plans/2026-10-14-personal-kb-revival.md) | **活跃** | P0（检索质量、Docker）与 P1（真流式、溯源、前端错误处理）已交付；P2（滚动摘要、评估收尾、知识库管理、解析增强）待排期 |
| [旧 P0 实施计划](superpowers/plans/2026-09-10-p0-agent-core-foundation.md) | 部分交付 | Agent Core 重构的实施记录。Task 1–7 已交付，Task 8（真流式）已由新计划吸收；**Task 9（滚动增量摘要）与 Task 10（评测收尾）仍未做，规格现成可直接用**。该文档同时保留了每个任务「原始规范 vs 实际落地」的差异表——**里面的代码块是原始规范，不要照抄** |
| [GSAP 动画设计](superpowers/specs/2026-06-03-gsap-animations-design.md) | 已实现 | 前端动画体系（"Print Impression" 语言）的设计说明，改动效时参考 |

根目录的 `CLAUDE.md` 是给 AI 协作者看的工程笔记：架构、并发模型、检索栈语义、测试分级、
环境硬前提。改架构前先读它。

## 阅读顺序建议

1. `CLAUDE.md` —— 这个仓库是怎么运转的、哪里有坑
2. `docs/tech-notes/verified-facts.md` —— 具体哪一行有问题
3. `docs/plans/2026-10-14-personal-kb-revival.md` —— 接下来做什么