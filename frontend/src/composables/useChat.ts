import { ref } from 'vue'
import { useChatStore } from '../stores/chat'
import { useConversations } from './useConversations'
import type { ChatMessage, Source } from '../types'

export function useChat() {
  const store = useChatStore()
  const isStreaming = ref(false)
  const currentRoute = ref('')
  const streamingContent = ref('')
  const streamError = ref('')
  const { fetchConversations, loadConversation } = useConversations()

  async function sendMessage(question: string, history: any[]) {
    store.addMessage({ role: 'user', content: question })
    isStreaming.value = true
    streamingContent.value = ''
    streamError.value = ''
    currentRoute.value = '思考中'

    let finalAnswer = ''
    let roundSources: Source[] = []

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: store.conversationId,
          question,
          history,
        }),
      })

      // 非 2xx 时响应体是后端返回的 JSON 错误，不是 SSE。这里必须挡住，
      // 否则会把 {"detail": ...} 当成 SSE 帧解析，或直接抛在下面的 reader 上。
      if (!response.ok) {
        throw new Error(`请求失败（HTTP ${response.status}）`)
      }
      if (!response.body) {
        throw new Error('响应没有可读流')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue

          let data: any
          try {
            data = JSON.parse(line.slice(6))
          } catch {
            continue // 半截帧或非 JSON 心跳，忽略
          }

          // 按 payload 的 type 分发，而不是按 data.answer / data.content 猜形状 ——
          // 旧实现靠猜，导致 done 事件里的 route 永远取不到（if data.answer 先命中）。
          switch (data.type) {
            case 'token':
              streamingContent.value += data.text
              break
            case 'tool_call':
              currentRoute.value = `调用工具：${data.name}`
              break
            case 'tool_result':
              currentRoute.value = data.ok ? `工具返回（${data.ms}ms）` : `工具失败：${data.error}`
              break
            case 'citation':
              roundSources.push(toSource(data))
              break
            case 'final':
              finalAnswer = data.answer
              break
            case 'error':
              streamError.value = data.message || '未知错误'
              break
            default:
              // 非协议事件：status（处理中）与 done（携带 conversation_id）
              if (data.status === 'thinking') currentRoute.value = '思考中'
              if (data.conversation_id && !store.conversationId) {
                store.setConversationId(data.conversation_id)
              }
          }
        }
      }

      if (finalAnswer) {
        // 先清空流式气泡再落定消息：反过来会出现「逐字内容 + 完整消息」同时存在，
        // 看起来像答案被渲染了两遍。
        streamingContent.value = ''
        store.addMessage({
          role: 'assistant',
          content: finalAnswer,
          sources: roundSources.length ? roundSources : undefined,
          sourceType: roundSources.length ? 'doc' : undefined,
        })
      } else if (!streamError.value) {
        streamError.value = '没有收到任何回答内容'
      }
    } catch (err) {
      streamError.value = err instanceof Error ? err.message : String(err)
    } finally {
      // 复位必须在 finally：任何异常路径（网络中断、后端 500、解析失败）都会跳过
      // 原来的收尾代码，把输入框永久留在 disabled 状态，只能刷新页面。
      isStreaming.value = false
      currentRoute.value = ''
      streamingContent.value = ''
      fetchConversations()
    }
  }

  async function loadConversationHistory(id: number) {
    const detail = await loadConversation(id)
    const chatMessages: ChatMessage[] = detail.messages.map(m => ({
      role: (m.role as 'user' | 'assistant'),
      content: m.content,
      sourceType: (m.source_type as 'web' | 'doc') || undefined,
      // 引用随历史一起回来：只在本次会话里显示来源，等于关掉页面就没了
      sources: m.sources?.length ? m.sources.map(toSource) : undefined,
    }))
    store.loadMessages(chatMessages)
    store.setConversationId(id)
  }

/** 后端引用事件 → 前端来源条目。文档片段没有 URL，只有来源与摘要。 */
function toSource(raw: any): Source {
  return {
    title: raw.source || '未知来源',
    link: '',
    snippet: raw.chunk,
  }
}

  function clearMessages() {
    streamError.value = ''
    store.clearMessages()
  }

  return {
    messages: store.messages,
    isStreaming,
    currentRoute,
    streamingContent,
    streamError,
    sendMessage,
    loadConversationHistory,
    clearMessages,
  }
}