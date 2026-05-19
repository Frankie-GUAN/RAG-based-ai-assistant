<template>
  <div class="flex flex-col h-full">
    <header class="px-6 py-4 border-b border-gray-100 bg-white flex items-center justify-between">
      <h2 class="text-lg font-semibold text-gray-800">对话</h2>
      <button @click="clearMessages" class="text-xs text-gray-400 hover:text-gray-600 transition-colors">清空对话</button>
    </header>

    <div class="flex-1 overflow-y-auto px-6 py-4" ref="msgContainer">
      <ChatMessage v-for="(msg, i) in messages" :key="i" :message="msg" />
      <AgentThinking :route="currentRoute" :is-streaming="isStreaming" />
      <div v-if="streamingContent" class="flex justify-start py-3">
        <div class="max-w-[75%] rounded-xl px-4 py-3 bg-white border border-gray-200 text-sm">
          {{ streamingContent }}
        </div>
      </div>
    </div>

    <div class="p-4 border-t border-gray-100 bg-white">
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

async function handleSend(text: string) {
  const history = messages.slice(-6).map(m => ({ role: m.role, content: m.content }))
  await sendMessage(text, history, false)
}
</script>
