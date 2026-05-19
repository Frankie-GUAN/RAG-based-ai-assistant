import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ChatMessage, ConversationSummary } from '../types'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const conversationId = ref<number | null>(null)
  const conversations = ref<ConversationSummary[]>([])
  const conversationsLoading = ref(false)

  function addMessage(msg: ChatMessage) {
    messages.value.push(msg)
  }

  function setConversationId(id: number | null) {
    conversationId.value = id
  }

  function loadMessages(msgs: ChatMessage[]) {
    messages.value.splice(0, messages.value.length, ...msgs)
  }

  function setConversations(list: ConversationSummary[]) {
    conversations.value.splice(0, conversations.value.length, ...list)
  }

  function removeConversation(id: number) {
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx !== -1) conversations.value.splice(idx, 1)
  }

  function prependConversation(summary: ConversationSummary) {
    conversations.value.unshift(summary)
  }

  function clearMessages() {
    messages.value.splice(0, messages.value.length)
    conversationId.value = null
  }

  return {
    messages, conversationId, conversations, conversationsLoading,
    addMessage, setConversationId, loadMessages, setConversations,
    removeConversation, prependConversation, clearMessages,
  }
})
