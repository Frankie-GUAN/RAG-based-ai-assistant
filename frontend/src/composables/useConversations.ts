import { useChatStore } from '../stores/chat'
import type { ConversationDetail } from '../types'

export function useConversations() {
  const store = useChatStore()

  async function fetchConversations() {
    store.conversationsLoading = true
    try {
      const res = await fetch('/api/conversations')
      const data = await res.json()
      store.setConversations(data)
    } finally {
      store.conversationsLoading = false
    }
  }

  async function loadConversation(id: number): Promise<ConversationDetail> {
    const res = await fetch(`/api/conversations/${id}`)
    if (!res.ok) throw new Error(`Failed to load conversation ${id}`)
    return res.json()
  }

  async function renameConversation(id: number, title: string) {
    await fetch(`/api/conversations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    })
  }

  async function deleteConversation(id: number) {
    await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
    store.removeConversation(id)
  }

  return { fetchConversations, loadConversation, renameConversation, deleteConversation }
}
