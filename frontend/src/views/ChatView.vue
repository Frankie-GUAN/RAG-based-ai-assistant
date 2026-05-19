<template>
  <div class="flex flex-col h-full">
    <!-- Header -->
    <header class="shrink-0 px-6 py-4 flex items-center justify-between"
      style="background: var(--ink-soft); border-bottom: 1px solid var(--border)">
      <h2>对话</h2>
      <button @click="clearMessages" class="text-xs transition-colors hover:text-[var(--page)]" style="color: var(--page-faint)">
        清空对话
      </button>
    </header>

    <!-- Messages -->
    <div class="flex-1 overflow-y-auto px-6 py-6" ref="msgContainer">
      <div v-if="messages.length === 0" class="flex flex-col items-center justify-center h-full text-center">
        <div class="text-5xl mb-4 opacity-30">◈</div>
        <p class="text-lg mb-2" style="color: var(--page-dim)">开始一段对话</p>
        <p class="text-sm max-w-xs" style="color: var(--page-faint)">Agentic RAG 结合本地文档检索与 DeepSeek 大模型，提供精准的智能问答</p>
      </div>

      <transition-group name="msg" tag="div">
        <ChatMessage v-for="msg in messages" :key="messages.indexOf(msg)" :message="msg" />
      </transition-group>

      <AgentThinking :route="currentRoute" :is-streaming="isStreaming" />

      <!-- Streaming preview -->
      <div v-if="streamingContent" class="flex justify-start py-2">
        <div class="max-w-[72%] rounded-xl px-4 py-3 text-sm leading-relaxed"
          style="background: var(--ink-muted); border: 1px solid var(--border); color: var(--page)">
          {{ streamingContent }}<span class="inline-block w-1 h-4 ml-0.5 animate-pulse align-middle" style="background: var(--brass)"></span>
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="shrink-0 p-4" style="background: var(--ink-soft); border-top: 1px solid var(--border)">
      <ChatInput :disabled="isStreaming" @send="handleSend" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { useChat } from '../composables/useChat'
import ChatMessage from '../components/chat/ChatMessage.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import AgentThinking from '../components/chat/AgentThinking.vue'

const { messages, isStreaming, currentRoute, streamingContent, sendMessage, clearMessages } = useChat()
const msgContainer = ref<HTMLElement>()

watch(messages, async () => {
  await nextTick()
  if (msgContainer.value) {
    msgContainer.value.scrollTop = msgContainer.value.scrollHeight
  }
}, { deep: true })

watch(streamingContent, async () => {
  await nextTick()
  if (msgContainer.value) {
    msgContainer.value.scrollTop = msgContainer.value.scrollHeight
  }
})

async function handleSend(text: string) {
  const history = messages.slice(-6).map(m => ({ role: m.role, content: m.content }))
  await sendMessage(text, history, false)
}
</script>

<style scoped>
.msg-enter-active {
  transition: opacity 300ms ease, transform 300ms ease;
}
.msg-enter-from {
  opacity: 0;
  transform: translateY(8px);
}
</style>
