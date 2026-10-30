# 前端

Vue 3 + TypeScript + Vite + Pinia + Tailwind CSS v4。

## 命令

```bash
npm install
npm run dev       # 开发服务器，默认 http://localhost:5173
npm run build     # vue-tsc 类型检查 + vite 构建到 dist/
npm run preview   # 预览构建产物
```

## 结构

```
src/
├── components/
│   ├── chat/         # 对话界面（消息流、Agent 思考过程、来源标签）
│   ├── knowledge/    # 知识库管理（上传、列表）
│   └── evaluation/   # 评测看板
├── composables/      # 所有网络 I/O 都在这里（useChat / useConversations / useKnowledge / useEvaluation）
├── stores/           # Pinia，只存状态，不发请求
├── views/            # 路由页面
└── router/
```

## 两条约定

**1. 状态与 I/O 分离**：`stores/` 只存状态，请求全写在 `composables/` 里。这样状态可以单独测试，请求逻辑也集中。

**2. SSE 按 `payload.type` 分发，不靠数据形状猜**：`useChat.ts` 用 `fetch` + `ReadableStream` 消费（不是 `EventSource`，因为请求是 POST）。事件类型见后端 `app/agent/protocol.py`——`token` / `tool_call` / `tool_result` / `citation` / `plan` / `reflect` / `cache_hit` / `final` / `error`。未知类型走 `default` 分支容错，后端加事件不会让前端崩。

流式收尾放在 `finally`：任何异常（网络中断、非 2xx、坏帧）都必须复位 `isStreaming`，否则输入框会永久禁用。

## 已知不足

无单元测试、无 lint。所有请求用裸 `fetch` 硬编码 `/api/...` 相对路径（`axios` 是依赖但没被使用）。base URL 只存在于 `vite.config.ts`（开发代理）和 `nginx.conf`（生产）两处。
