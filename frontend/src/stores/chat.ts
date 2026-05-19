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
    messages.value = msgs
  }

  function setConversations(list: ConversationSummary[]) {
    conversations.value = list
  }

  function removeConversation(id: number) {
    conversations.value = conversations.value.filter(c => c.id !== id)
  }

  function prependConversation(summary: ConversationSummary) {
    conversations.value.unshift(summary)
  }

  function clearMessages() {
    messages.value = []
    conversationId.value = null
  }

  return {
    messages, conversationId, conversations, conversationsLoading,
    addMessage, setConversationId, loadMessages, setConversations,
    removeConversation, prependConversation, clearMessages,
  }
})
