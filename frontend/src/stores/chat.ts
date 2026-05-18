import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ChatMessage } from '../types'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const conversationId = ref<number | null>(null)

  function addMessage(msg: ChatMessage) {
    messages.value.push(msg)
  }

  function clearMessages() {
    messages.value = []
    conversationId.value = null
  }

  return { messages, conversationId, addMessage, clearMessages }
})
