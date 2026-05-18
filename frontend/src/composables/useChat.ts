import { ref } from 'vue'
import type { ChatMessage } from '../types'

export function useChat() {
  const messages = ref<ChatMessage[]>([])
  const isStreaming = ref(false)
  const currentRoute = ref('')
  const streamingContent = ref('')

  async function sendMessage(question: string, history: any[], hasDocs: boolean) {
    messages.value.push({ role: 'user', content: question })
    isStreaming.value = true
    streamingContent.value = ''

    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history, has_docs: hasDocs }),
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
          if (data.status === 'thinking') {
            currentRoute.value = '思考中...'
          } else if (data.route) {
            currentRoute.value = data.route
          } else if (data.content) {
            streamingContent.value += data.content
          } else if (data.answer) {
            messages.value.push({
              role: 'assistant',
              content: data.answer,
              sourceType: data.route === 'web' ? 'web' : data.route === 'rag' ? 'doc' : undefined,
            })
          }
        }
      }
    }

    isStreaming.value = false
    currentRoute.value = ''
    streamingContent.value = ''
  }

  function clearMessages() {
    messages.value = []
  }

  return { messages, isStreaming, currentRoute, streamingContent, sendMessage, clearMessages }
}
