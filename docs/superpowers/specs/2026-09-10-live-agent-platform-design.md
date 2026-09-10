# PRD：端云协同的直播主播经营 Agent 平台

- **日期**：2026-09-10
- **状态**：已确认，待实施
- **目标岗位**：字节跳动 · Agent 开发工程师 - 抖音直播
- **项目基线**：现有 `Agentic RAG 智能问答平台`（FastAPI + LangGraph + Vue 3）
- **周期**：保底 3.5 周 / 简历可用 8 周 / 完整 12 周（详见 §6.0）

---

## 1. 背景与问题

### 1.1 现状

现有项目是一个通用的 Agentic RAG 文档问答系统，技术上包含：

- LangGraph 固定 DAG 编排（`router → retrieve/web_search → generate`）
- 自建工具注册表（`ToolRegistry`，dict 查找）
- BGE-M3 稠密检索 + BM25 稀疏检索 + RRF 融合 + BGE-Reranker-v2-m3 精排
- MySQL 持久化会话，LLM 摘要压缩
- Vue 3 前端 + SSE 推送
- 手写 LLM-as-Judge 评测（忠实度 + 答案相关性）

### 1.2 核心问题

**问题一：README 宣称的能力与代码不符（最高优先级）**

经代码核查确认：

| README 宣称 | 代码实际 | 证据 |
|---|---|---|
| "MCP 协议：标准化工具接口" | 无任何 MCP 依赖或实现 | `grep -rin "mcp" backend/` 零命中；`app/tools/registry.py` 是纯 dict 查找 |
| "SSE 流式输出：实时推送，逐 token 显示" | 假流式 | `grep -n "astream\|stream=True"` 零命中；[chat.py:73-78](../../../backend/app/api/chat.py#L73-L78) 等 LLM 全量返回后再切 4 字符分片 |

这两点是面试中的**致命风险**：面试官打开 `backend/app/tools/` 目录即可识破。

**问题二：Agent 名不副实**

`app/agent/graph.py` 是三节点线性 DAG，`AgentState.iteration` 字段有定义但从未被读取。系统没有 agent loop、没有工具调用决策、没有规划、没有反思——本质是"带路由的 RAG pipeline"，不是 Agent。

**问题三：与目标岗位的能力缺口**

| JD 要求 | 现状 | 缺口 |
|---|---|---|
| 客户端 Agent 框架研发 | 前端为纯展示层 | 完全缺失 |
| 感知·理解·规划·执行 | 单跳路由 | 缺规划、反思、多步执行 |
| 端侧、云端协同 | 无 | 完全缺失 |
| 工具调用 / 执行链路优化 | 同步串行调用，无并行/重试 | 缺并行、重试、降级 |
| Multi-Agent | 无 | 完全缺失 |
| Skills | 无 | 完全缺失 |
| 长短期记忆 | MySQL 消息 + LLM 摘要 | 缺向量化长期记忆、用户画像 |
| 任务规划 | 无 | 完全缺失 |
| 评测 | 2 个指标，串行 `sleep(3)` 阻塞 | 缺检索指标、Agent 指标、Golden Set、CI |

此外存在**死依赖**：`langchain-google-genai`、`google-generativeai` 是早期使用 Gemini 时引入、切换到 DeepSeek 后未清理的遗留依赖，以及 `requirements.txt` 中声明但代码从未使用的 `ragas`。

### 1.3 目标

把项目重构为 **「端云协同的直播主播经营 Agent 平台」**：

1. **兑现承诺**——把 README 里已经宣称的 MCP、流式真正实现，消除面试风险
2. **做深内核**——从固定 DAG 升级为具备规划、执行、反思闭环的真实 Agent
3. **补齐纵深**——Multi-Agent 编排、Skills 体系、长短期记忆、四层评测
4. **差异化**——端侧 TS Agent Runtime + 端云任务路由，正面回应"客户端 Agent 框架"与"端云协同"

**非目标**（YAGNI）：

- 不做真移动端（Flutter / 原生），端侧限定在浏览器
- 不做多租户、权限体系、计费
- 不做真实抖音数据接入，场景数据全部合成并以 mock 标注
- 不重写现有 RAG 检索链路（BGE-M3 + BM25 + RRF + Reranker 保留）

---

## 2. 目标用户与场景

**用户**：抖音直播的中小主播及其运营人员。

**核心痛点**：

| 痛点 | 现有人工方式 | Agent 平台解法 |
|---|---|---|
| 平台规则复杂，违规红线难记 | 翻文档 / 问运营 / 踩坑后才知道 | 规则合规 Agent，带引用溯源 |
| 商品话术撰写低效且易踩合规线 | 手写 + 事后被罚 | 话术生成 Agent + 强制合规校验 |
| 直播数据看不懂，不知问题在哪 | 看后台图表靠经验猜 | 数据诊断 Agent，多步归因 |
| 复盘靠人肉翻数据，耗时数小时 | 手动导表 + Excel | 复盘报告 Agent，一键生成 |
| 弹幕违规反应滞后，等发现已违规 | 人工盯屏 | 端侧实时风控，P95 < 200ms |

**关键场景**（对应第 6 节功能清单）：

1. 开播前：查规则、备话术、看商品
2. 开播中：实时弹幕风控、商品答疑
3. 开播后：数据诊断、复盘报告

---

## 3. 目标架构

### 3.1 架构决策

**决策一：云端保留 LangGraph，端侧自研 TS Agent Core，双端共享统一 Agent 协议。**

理由：LangGraph 是 Python 库，无法运行在浏览器中，因此端侧必须自研。而"负责基于大语言模型的客户端 Agent 框架研发"正是 JD 第 1 条的字面要求——自研端侧 runtime 是能力证明，不是负担。云端保留 LangGraph 则是复用既有资产并保持技术栈主流性。

**决策二：协议先行。** 先定义 `AgentEvent` 事件流契约，两端各自实现，避免维护两套世界观。

### 3.2 架构图

```
┌──────────────────────── 浏览器端 (Vue 3 + Web Worker) ────────────────────────┐
│                                                                              │
│  UI 层        ChatView · TraceView · DashboardView · EvaluationView          │
│  ─────────────────────────────────────────────────────────────────────────   │
│  Edge Agent Runtime  (TypeScript, Web Worker)          ← 本项目核心自研      │
│   ├─ Intent 快判 (规则 + 轻量 embedding)                                     │
│   ├─ Local Retriever (IndexedDB / OPFS 向量存储)                             │
│   ├─ Agent Loop (tool-calling, 轻量子集)                                     │
│   ├─ 本地工具执行器 (弹幕流监听 / 页面上下文 / 缓存读写)                      │
│   ├─ Edge-Cloud Router                                                       │
│   └─ Offline Fallback                                                        │
│  ─────────────────────────────────────────────────────────────────────────   │
│  Agent Protocol (统一事件流契约，Python / TS 双端实现)                       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ SSE / WebSocket
┌───────────────────────────────▼──────────────────────────────────────────────┐
│                       云端 (FastAPI + LangGraph)                              │
│                                                                              │
│  API 层    /chat/stream (真流式) · /trace · /evaluation · /knowledge          │
│  ──────────────────────────────────────────────────────────────────────────  │
│  编排层   Supervisor (Planner → Executor → Reflector → Replanner)             │
│    ├─ 规则合规 Agent                                                          │
│    ├─ 数据诊断 Agent                                                          │
│    ├─ 话术生成 Agent                                                          │
│    └─ 复盘报告 Agent                                                          │
│  ──────────────────────────────────────────────────────────────────────────  │
│  基建层   上下文工程引擎 · Skill Registry · Memory Manager · MCP Tool Layer   │
│  ──────────────────────────────────────────────────────────────────────────  │
│  RAG      BGE-M3 + BM25 + RRF + BGE-Reranker                                 │
│  可观测   Trace Collector (thought / tool / result 全链路落库)                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 模块边界

每个模块单一职责、接口明确、可独立测试：

| 模块 | 职责 | 依赖 | 对外接口 |
|---|---|---|---|
| `agent/protocol.py` / `protocol.ts` | 定义 AgentEvent 契约与序列化 | 无 | 类型定义 |
| `agent/core`（双端） | 执行 agent loop | protocol, tools | `run(task, ctx) -> AsyncIterable<AgentEvent>` |
| `agent/planner` | 任务分解、重规划 | LLM | `plan(goal, ctx) -> PlanStep[]` |
| `agent/reflector` | 自评与重试决策 | LLM | `reflect(trace) -> Verdict` |
| `agent/supervisor` | 专家 Agent 调度与仲裁 | 各专家 Agent | `dispatch(task) -> AgentEvent` |
| `skills/` | Skill 包注册与按需注入 | 文件系统 | `SkillRegistry.load(name)` |
| `context/engine` | 分层 token 预算与压缩 | tokenizer | `assemble(sources) -> Messages` |
| `memory/` | 短期/长期记忆读写 | 向量库, MySQL | `recall(query) -> Entry[]` / `write(entry)` |
| `edge/router` | 端云任务路由决策 | 无（纯函数） | `decide(task) -> 'edge' \| 'cloud'` |
| `edge/runtime` | 端侧 Agent 执行 | protocol, 端侧工具 | 同 `agent/core` |
| `mcp/server` | 以 MCP 协议暴露工具 | mcp SDK | MCP tool 定义 |
| `observability/trace` | 采集与回放执行链路 | MySQL | `TraceStore` |

### 3.4 数据流（一次典型请求）

```
用户提问 "上周哪场直播转化率最低，为什么？"
  │
  ├─ 端侧：意图快判 → 判定为"数据诊断类，需多步推理" → 复杂度高 → 路由到云端
  │        emit { type: 'handoff', from: 'edge', to: 'cloud', reason: 'complex-planning' }
  │
  ├─ 云端 Supervisor：Planner 生成 DAG
  │        emit { type: 'plan', steps: [查场次数据, 取话术记录, 对比基准, 归因] }
  │
  ├─ Executor：step1、step2 无依赖 → asyncio.gather 并行调用
  │        emit { type: 'tool_call', exec: 'cloud' } × 2
  │        emit { type: 'tool_result', ms: 340 } × 2
  │
  ├─ step3 依赖前两步结果 → 串行执行
  ├─ step4 交给数据诊断 Agent 归因
  │
  ├─ Reflector：检查答案是否有数据支撑、引用是否可溯 → pass
  │        emit { type: 'reflect', verdict: 'pass' }
  │
  ├─ 真流式输出 → emit { type: 'token' } × N
  ├─ 引用溯源   → emit { type: 'citation', docId, score }
  ├─ 记忆写入   → emit { type: 'memory_write', scope: 'long_term' }
  └─ emit { type: 'final', answer, citations, usage }
```

---

## 4. 核心设计

### 4.1 统一 Agent 协议

整个改造的技术地基。Python 与 TypeScript 双端实现同一契约。

```ts
type AgentEvent =
  // 规划
  | { type: 'plan',          steps: PlanStep[] }
  // 推理过程
  | { type: 'thought',       text: string, stepId: string }
  // 工具调用与结果（exec 标记执行位置）
  | { type: 'tool_call',     id: string, name: string, args: object, exec: 'edge' | 'cloud' }
  | { type: 'tool_result',   id: string, ok: boolean, data: unknown, ms: number, error?: string }
  // 真流式 token
  | { type: 'token',         text: string }
  // 引用溯源
  | { type: 'citation',      docId: string, chunk: string, score: number, source: string }
  // 记忆写入
  | { type: 'memory_write',  scope: 'session' | 'long_term', entry: MemoryEntry }
  // 端云交接
  | { type: 'handoff',       from: 'edge' | 'cloud', to: 'edge' | 'cloud',
                             reason: string, latencyBudget?: number }
  // 反思结论
  | { type: 'reflect',       verdict: 'pass' | 'retry', critique: string }
  // 终态
  | { type: 'final',         answer: string, citations: Citation[], usage: Usage }
  // 结构化错误
  | { type: 'error',         code: string, message: string, recoverable: boolean }
```

**设计价值**：

1. 前端可**回放整个 Agent 执行过程**（TraceView），这是面试中的关键差异化
2. 端云交接有明确语义（`handoff` + `latencyBudget`）
3. 端侧 runtime 是云端协议的子集，无需维护两套概念模型

### 4.2 Agent 内核

**现状**：`router → retrieve → generate` 三节点线性。

**目标**：完整闭环。

```
Perceive   意图分类 + 槽位抽取
           （直播域槽位：主播ID / 商品ID / 时间窗 / 指标类型）
   ↓
Understand 指代消解（"那场直播" → 具体场次 ID）
           话题漂移检测（与当前会话主题的相似度低于阈值 → 重置上下文）
           歧义澄清（槽位缺失且无法推断 → 主动追问）
   ↓
Plan       复杂任务分解为 DAG
           （"上周哪场转化率最低，为什么"
             → [查场次数据] → [取话术记录] → [对比基准] → [归因分析]）
   ↓
Execute    tool-calling loop
           · 无依赖步骤 asyncio.gather 并行
           · 失败重试（指数退避，最多 3 次）
           · 超时降级
   ↓
Reflect    自评：答案有据吗？引用对得上吗？
           不达标 → Replan（最多 2 轮）
   ↓
Respond    真流式输出 + 引用溯源
```

**关键工程约束**：

| 约束 | 规则 |
|---|---|
| 循环终止 | `iteration` 上限 8；token 预算上限；**无进展检测**（连续 2 轮无新信息即停） |
| 工具失败 | 返回结构化 `error` 写入 scratchpad，由 LLM 决策换工具或降级回答；工具失败 ≠ 任务失败 |
| 并行边界 | 仅无数据依赖的步骤并行；有依赖的严格串行 |
| 重试策略 | 指数退避 1s / 2s / 4s；限流类错误（429）额外退避 |

### 4.3 Multi-Agent 编排

```
Supervisor（云端强模型：规划 + 仲裁 + 结果聚合）
├─ 规则合规 Agent   skills: 平台规则库 · 违规检测 · 引用溯源
├─ 数据诊断 Agent   skills: 数据查询 · 同环比 · 归因分析
├─ 话术生成 Agent   skills: 商品库 · 话术模板 · 合规校验
└─ 复盘报告 Agent   skills: 聚合前三者输出 → 结构化报告
```

**设计要点**：

- Supervisor 只做调度与仲裁，不直接执行工具（避免职责膨胀）
- 专家 Agent 之间不直接通信，一律经 Supervisor 中转（可观测、可仲裁）
- 话术生成 Agent 的产出**必须**过合规校验 Skill，校验失败则重生成（最多 2 次）

### 4.4 Skill 体系

JD 明确点名 Skills，需做成规范的可复用包：

```
skills/compliance-check/
├── SKILL.md          # 名称 / 描述 / 触发条件（给 Supervisor 做路由决策）
├── prompt.md         # 该 Skill 的 system prompt 片段
├── tools.yaml        # 依赖的 MCP 工具及参数约束
├── examples.jsonl    # few-shot 示例
└── validators.py     # 输出校验（如"必须返回引用 ID"）
```

**Skill 按需注入上下文**——这是上下文工程的具体抓手，而非空泛概念。Supervisor 依据 `SKILL.md` 的触发条件决定加载哪些 Skill，未加载的 Skill 不占用 token。

**首期交付 3 个 Skill**：`compliance-check`（合规校验）、`data-attribution`（数据归因）、`script-generation`（话术生成）。

### 4.5 记忆体系

| 层 | 存储 | 内容 | 生命周期 |
|---|---|---|---|
| Scratchpad | 内存 | 当前任务的中间工具结果 | 单次任务；超预算即 LLM 压缩为"中间结论" |
| 会话记忆 | MySQL + 摘要 | 最近 N 轮 + 滚动摘要（**已实现，保留**） | 会话级 |
| Episode 记忆 | 向量库 | 历史相似问题 + 成功解决路径 | 长期；检索 top-k 注入 |
| 主播画像 | MySQL 结构化 | 类目 / 客单价 / 历史违规 / 话术风格偏好 | 长期；个性化依据 |

**Episode 记忆的价值**：新问题先检索"历史上类似问题是如何解决的"，命中则复用工具调用路径。这既是个性化（契合 JD 的"个性化服务"），也是**执行链路优化**（省 token、省步数、降延迟）。

### 4.6 端云协同

**端侧 Runtime 职责**（Web Worker，TypeScript）：

- 本地向量检索（IndexedDB / OPFS + 端侧 embedding）
- 意图快判、槽位抽取、高频问题直答
- 端侧工具：弹幕流监听、页面上下文采集、本地缓存读写
- 离线降级：断网时基于缓存规则库 / 话术库仍可服务

**端侧模型选型**：浏览器无法运行 BGE-M3（体积与算力均不现实）。端侧采用 `bge-small-zh` 的 ONNX 量化版（约 25MB），或在不满足条件时退化为「关键词匹配 + 缓存向量相似度」。选型在 P2 首日实测体积与延迟后定稿。

**Edge-Cloud Router 决策函数**：

```
cost = w₁ · 隐私敏感度
     + w₂ · (1 − 端侧置信度)
     + w₃ · 任务复杂度
     + w₄ · 延迟紧迫度

cost > θ  → 上云       cost ≤ θ → 端侧执行
```

**分流策略表**：

| 任务 | 端侧 | 云端 | 理由 |
|---|:---:|:---:|---|
| 已知违规话术命中 | ✅ | — | 本地向量库，< 50ms |
| 弹幕实时风控首轮 | ✅ | — | 100 条/秒，必须端侧 |
| 风控模糊 case 复核 | — | ✅ | 需大模型判断 |
| 意图快判 / 槽位抽取 | ✅ | — | 隐私 + 低延迟 |
| 多步数据归因 | — | ✅ | 需强推理 |
| 话术生成 | — | ✅ | 需大模型 + 合规校验 |
| 规则库查询 | ✅ TTL 内 | ✅ TTL 过期时 | 时效性权衡 |

**数据回流闭环**（"协同"的实义，不只是分流）：

```
云端确认的违规样本 → 回写端侧向量库 → 端侧命中率提升 → 更多 case 无需上云
```

该闭环使端侧"越用越准"，同时降低云端成本与延迟，且离线可处理的 case 持续增多。

### 4.7 上下文工程引擎

分层 token 预算模型：

```
总预算 32k tokens
├─ System + Skill 定义        15%   固定
├─ 长期记忆 (Episode top-k)   10%   动态
├─ 短期会话 (最近 N 轮)        20%   动态
├─ 检索上下文 (reranked)       35%   动态
└─ Scratchpad (工具中间结果)   20%   动态，超预算 → LLM 压缩
```

**淘汰策略**：层内按「优先级 + LRU」淘汰；每层超限时先压缩再截断，不跨层抢占。

**度量指标**：上下文利用率、压缩触发率、因超限导致的截断次数。

### 4.8 工具层（真 MCP）

**现状问题**：`app/tools/registry.py` 是自建 dict 注册表，README 却宣称"MCP 协议"。

**目标**：使用官方 `mcp` Python SDK 实现真正的 MCP Server，将现有 3 个工具（`search_documents` / `search_web` / `parse_document`）与新增直播域工具统一以 MCP 协议暴露，Agent 侧作为 MCP Client 调用。

**改造要点**：

1. 保留 `ToolRegistry` 作为进程内缓存层，但其数据来源改为 MCP Server 的工具列表
2. 工具 schema 由 MCP 协议自动生成（移除手写的 `TOOL_SCHEMAS`）
3. 支持并行调用：无依赖的 tool_call 并发执行
4. 统一结构化错误返回，供 Agent 决策重试或降级

**新增直播域工具**：`query_live_sessions`（查询场次）、`query_metrics`（指标数据）、`search_products`（商品库）、`check_compliance`（合规检测）。

### 4.9 评测体系 v2

**现状**：仅 faithfulness + answer_relevancy 两个指标；串行执行且每次 `sleep(3)` 阻塞；`ragas` 已声明依赖但未使用。

**目标**：四层指标体系。

| 层级 | 指标 | 方法 |
|---|---|---|
| 检索 | Hit Rate@k · MRR · NDCG · Context Precision · Context Recall | Golden Set 人工标注 |
| 生成 | Faithfulness · Answer Relevancy · **引用准确率** | LLM-as-Judge |
| Agent | 任务成功率 · 工具选择准确率 · 平均步数 · 规划有效率 | 端到端任务集 |
| 端云 | 端侧命中率 · 路由准确率 · P50/P95 延迟 · 单次成本 | Trace 埋点 |

**三个必须完成的前提动作**（否则评测不可信）：

1. **Golden Set**：标注 50–100 条问题，覆盖四类意图（应检索 / 应直答 / 应联网 / 应调用工具）并包含对抗样本（无答案问题、歧义问题、诱导违规问题）
2. **LLM-Judge 校准**：人工标注 20 条作为基准，计算 Judge 与人工的一致率并在报告中披露——证明 Judge 结果可信
3. **CI 回归**：GitHub Actions 在每次 PR 上运行评测，与 baseline 对比 delta，指标下降则阻断合并

**性能改造**：将串行 `sleep(3)` 改为 `asyncio` 并发 + 信号量限流（并发度 4），消除阻塞。

---

## 5. 场景化：主播经营助手

### 5.1 功能清单

| 功能 | 编排方式 | 用到的能力 |
|---|---|---|
| 平台规则问答 | 规则合规 Agent | RAG + 引用溯源 |
| 商品卖点 / 话术生成 | 话术生成 Agent | 生成 + 合规强制校验 |
| 直播数据诊断 | 数据诊断 Agent + Planner | 多步工具调用 + 归因 |
| 直播复盘报告 | 复盘 Agent 组合前三者 | Multi-Agent 协作 |
| 实时弹幕风控 | 端侧 Runtime | 端云协同 + 低延迟 |

### 5.2 数据方案

全部使用**合成数据**，并在 UI 与文档中明确标注为 mock：

| 数据集 | 规模 | 生成方式 |
|---|---|---|
| 平台规则库 | 200+ 条款，覆盖 8 个类目 | 参照公开平台规则结构合成 |
| 商品库 | 50 个 SKU，含卖点 / 成分 / 禁忌 | 合成 |
| 直播场次数据 | 30 天 × 每日 2 场 | 合成，含转化率 / 停留时长 / 互动率等指标 |
| 话术模板库 | 30 套 | 合成 |
| 违规话术样本 | 100 条正例 + 100 条对抗样本 | 合成 |
| Golden Set | 50–100 条问答对 | 人工标注 |

---

## 6. 迭代计划

### 6.0 工期现实性说明

各任务按「熟练开发者全职投入」估算，累计工作量约 **59.5 人日（≈ 12 周）**，超出最初的 8 周窗口约 50%。因此不把 8 周当作硬承诺，而是把 P0/P1/P2 当作**优先级分层**：

| 交付档位 | 范围 | 实际工期 |
|---|---|---|
| **保底档** | P0 全部 | 3.5 周 |
| **简历可用档** | P0 + P1 | 8 周 |
| **完整档** | P0 + P1 + P2 | 12 周 |

**若 8 周是硬期限**，P2 按下表降级为 MVP（约 3 周 → 1.5 周）：

| 任务 | 完整实现 | 8 周降级版 |
|---|---|---|
| 端侧 TS Runtime | 完整 agent loop + 本地检索 | 仅风控链路 + 端云路由（协议已定义，能力留 backlog） |
| 离线降级 | 断网可用 | backlog |
| 评测 v2 | 四层指标全量 | 检索 + 生成两层（Agent / 端云指标进 backlog） |
| Trace 可观测 UI | 完整回放 | 简版时间线（仅展示 plan + tool_call + 耗时） |

降级不影响 JD 核心叙事：端云协同的**协议、路由决策、数据回流闭环**三项保留，被砍的是工程完备度而非架构。

### P0（第 1–3.5 周）：兑现承诺，消灭硬伤

| # | 任务 | 工期 |
|---|---|---|
| 1 | 真流式 SSE（直通 LLM stream，移除 4 字符分片） | 2d |
| 2 | 真 MCP 工具层（MCP Server + Agent 作 Client） | 5d |
| 3 | Agent Core：ReAct + tool-calling loop + 并行调用 + 重试 | 7d |
| 4 | 上下文工程引擎 | 3d |
| 5 | 清理死依赖（移除 gemini 包；`ragas` 启用或删除） | 0.5d |
| | **小计** | **17.5d** |

### P1（第 4–8 周）：Agent 能力纵深

| # | 任务 | 工期 |
|---|---|---|
| 6 | Planner / Executor / Reflector / Replanner | 5d |
| 7 | Multi-Agent Supervisor + 4 个专家 Agent | 5d |
| 8 | Skill 体系 + 3 个 Skill 包 | 3d |
| 9 | 记忆体系（Episode 向量库 + 主播画像） | 4d |
| 10 | 主播经营场景：知识库 + mock 数据工具集 | 5d |
| | **小计** | **22d** |

### P2（第 9–12 周）：端云协同 + 评测闭环

| # | 任务 | 工期 |
|---|---|---|
| 11 | Agent 协议 + 端侧 TS Runtime（Web Worker） | 7d |
| 12 | Edge-Cloud Router + 端侧向量检索 + 离线降级 | 5d |
| 13 | 评测 v2：Golden Set + 四层指标 + CI 回归 | 5d |
| 14 | Trace 可观测 UI（执行链路回放） | 3d |
| | **小计** | **20d** |

**总计：59.5 人日**（P0 17.5 + P1 22 + P2 20）

---

## 7. 验收标准

| 维度 | 指标 | 目标 |
|---|---|---|
| 延迟 | 端侧风控 P95 | < 200ms |
| 延迟 | 首 token 延迟（云端） | < 800ms |
| 延迟 | 并行工具调用带来的链路提速 | ≥ 40%（对比串行基线） |
| 质量 | Golden Set 引用准确率 | > 90% |
| 质量 | Faithfulness | > 0.85 |
| 质量 | LLM-Judge 与人工一致率 | > 80% |
| Agent | 任务成功率（端到端任务集） | > 85% |
| Agent | 规划有效率 | > 75% |
| 成本 | 单次请求 token 成本（对比基线） | 下降 ≥ 30% |
| 工程 | CI 评测可复现 | 是 |
| 工程 | 无死依赖、无 README 与实现不符项 | 是 |

---

## 8. 交付物

| 交付物 | 位置 |
|---|---|
| 本 PRD | `docs/superpowers/specs/2026-09-10-live-agent-platform-design.md` |
| 实施计划 | `docs/superpowers/plans/` |
| **JD 能力映射表** | `docs/jd-mapping.md` |
| 评测报告（含 Judge 校准结果） | `docs/evaluation-report.md` |
| 架构决策记录（ADR） | `docs/adr/` |
| README 修订（与实现对齐） | `README.md` |

**`docs/jd-mapping.md` 是本项目的特殊交付物**：将 JD 每一条要求映射到项目中的具体文件与 commit，并附面试讲述要点。这直接回应 JD 第 3 条「能清晰阐述如何通过 Agent 解决实际业务问题」。

---

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| MCP SDK 与现有 LangChain 工具生态集成不畅 | P0 任务 2 延期 | 保留 `ToolRegistry` 作适配层；MCP 集成失败时可降级为"工具层"并同步修正 README 措辞 |
| 端侧 embedding 模型体积/性能不达标 | P2 任务 12 效果打折 | 设降级路径：关键词匹配 + 缓存向量；端侧定位为"快筛"，精度由云端复核兜底 |
| 8 周周期与课业冲突 | 交付不完整 | P0 必须先完成（消除面试风险）；P1/P2 按优先级交付，未完成项在文档中显式标注为 backlog |
| 合成数据缺乏真实分布 | 指标虚高、结论不可信 | 在评测报告中明确标注数据为合成；Golden Set 全部人工标注以保证质量 |
| LLM-as-Judge 不可靠 | 评测结论无效 | 强制做人工校准并披露一致率；一致率低于 80% 则改用规则+人工混合评审 |

---

## 10. 面试叙述骨架

1. **问题**：通用文档问答 Agent 无法体现客户端 Agent 与端云协同能力
2. **判断**：JD 看重的是 Agent 内功（规划/记忆/评测）+ 场景落地能力 + 端侧工程能力
3. **方案**：协议先行的双 Runtime 架构 —— 云端 LangGraph 重编排，端侧自研 TS Agent Core
4. **最难的点**：
   - 端云任务边界的判定（不是技术问题，是成本/延迟/隐私的权衡）
   - Agent 循环的终止条件（防止无限循环与 token 爆炸）
   - 评测的可信度（Judge 会骗人，必须校准）
5. **结果**：用第 7 节的数字说话
6. **反思**：哪些设计过度、哪些还不够、如果重做会怎么改
