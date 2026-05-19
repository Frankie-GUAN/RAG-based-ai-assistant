import { ref } from 'vue'
import { useChatStore } from '../stores/chat'
import { useConversations } from './useConversations'
import type { ChatMessage } from '../types'

export function useChat() {
  const store = useChatStore()
  const isStreaming = ref(false)
  const currentRoute = ref('')
  const streamingContent = ref('')
  const { fetchConversations, loadConversation } = useConversations()

  async function sendMessage(question: string, history: any[], hasDocs: boolean) {
    store.addMessage({ role: 'user', content: question })
    isStreaming.value = true
    streamingContent.value = ''

    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: store.conversationId,
        question,
        history,
        has_docs: hasDocs,
      }),
    })

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6))
          if (data.answer) {
            store.addMessage({
              role: 'assistant',
              content: data.answer,
              sourceType: data.route === 'web' ? 'web' : data.route === 'rag' ? 'doc' : undefined,
            })
          } else if (data.status === 'thinking') {
            currentRoute.value = '思考中...'
          } else if (data.content) {
            streamingContent.value += data.content
          } else if (data.route) {
            currentRoute.value = data.route
          }
          // Capture conversation_id from done event for new conversations
          if (data.conversation_id && !store.conversationId) {
            store.setConversationId(data.conversation_id)
          }
        }
      }
    }

    isStreaming.value = false
    currentRoute.value = ''
    streamingContent.value = ''

    // Refresh sidebar to show updated conversation
    fetchConversations()
  }

  async function loadConversationHistory(id: number) {
    const detail = await loadConversation(id)
    const chatMessages: ChatMessage[] = detail.messages.map(m => ({
      role: (m.role as 'user' | 'assistant'),
      content: m.content,
      sourceType: (m.source_type as 'web' | 'doc') || undefined,
    }))
    store.loadMessages(chatMessages)
    store.setConversationId(id)
  }

  function clearMessages() {
    store.clearMessages()
  }

  return {
    messages: store.messages,
    isStreaming,
    currentRoute,
    streamingContent,
    sendMessage,
    loadConversationHistory,
    clearMessages,
  }
}
