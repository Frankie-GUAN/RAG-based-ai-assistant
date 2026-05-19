<template>
  <div class="flex h-full">
    <!-- Main conversation column -->
    <div class="flex-1 flex flex-col min-w-0">
      <!-- Messages area — generous centered column -->
      <div class="flex-1 overflow-y-auto" ref="msgContainer">
        <div class="max-w-2xl mx-auto px-8 py-10">
          <!-- Empty state -->
          <div v-if="messages.length === 0" class="flex flex-col items-center justify-center" style="min-height: 60vh">
            <p class="text-5xl mb-6" style="font-family: var(--font-display); color: var(--ink)">
              What do you <span style="color: var(--clay); font-style: italic">need</span><br />to know today?
            </p>
            <p class="text-base max-w-md text-center leading-relaxed" style="color: var(--ink-muted)">
              Ask a question. The Agent will search your documents, the web, or reason directly.
            </p>
          </div>

          <!-- Messages -->
          <transition-group name="msg" tag="div" class="space-y-1">
            <ChatMessage v-for="(msg, idx) in messages" :key="idx" :message="msg" />
          </transition-group>

          <!-- Thinking -->
          <AgentThinking :route="currentRoute" :is-streaming="isStreaming" />

          <!-- Streaming content -->
          <div v-if="streamingContent" class="py-5">
            <p class="text-base leading-relaxed" style="color: var(--ink)">
              {{ streamingContent }}<span class="cursor-blink"></span>
            </p>
          </div>

          <!-- Bottom spacer for scroll comfort -->
          <div class="h-8"></div>
        </div>
      </div>

      <!-- Input area — floated above bottom with generous padding -->
      <div class="shrink-0 px-8 pb-8 pt-2">
        <div class="max-w-2xl mx-auto">
          <ChatInput :disabled="isStreaming" @send="handleSend" />
        </div>
      </div>
    </div>

    <!-- Context panel — slides in when there are sources -->
    <transition name="panel">
      <aside v-if="showContext" class="w-80 shrink-0 border-l overflow-y-auto px-5 py-10"
        style="background: var(--white-warm); border-color: var(--rule)">
        <h3 class="text-xs uppercase tracking-widest mb-5" style="color: var(--ink-muted); font-family: var(--font-body)">
          Sources &amp; Context
        </h3>
        <div class="space-y-4">
          <div v-for="(ctx, i) in contextItems" :key="i"
            class="text-sm leading-relaxed pb-4" style="border-bottom: 1px solid var(--rule-light)">
            <p class="font-semibold mb-1" style="color: var(--clay); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em">
              {{ ctx.label }}
            </p>
            <p style="color: var(--ink-soft)">{{ ctx.text }}</p>
          </div>
        </div>
        <button @click="showContext = false"
          class="mt-6 text-xs opacity-40 hover:opacity-70 transition-opacity">Close panel</button>
      </aside>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, computed } from 'vue'
import { useChat } from '../composables/useChat'
import ChatMessage from '../components/chat/ChatMessage.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import AgentThinking from '../components/chat/AgentThinking.vue'

const { messages, isStreaming, currentRoute, streamingContent, sendMessage } = useChat()
const msgContainer = ref<HTMLElement>()
const showContext = ref(false)

const contextItems = computed(() => {
  return messages
    .filter(m => m.role === 'assistant' && m.sources?.length)
    .flatMap(m => m.sources!.map(s => ({ label: m.sourceType === 'web' ? 'Web' : 'Document', text: s.snippet || s.title })))
    .slice(-10)
})

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
.msg-enter-active { transition: opacity 400ms ease, transform 400ms ease; }
.msg-enter-from { opacity: 0; transform: translateY(12px); }

.panel-enter-active, .panel-leave-active { transition: transform 300ms ease, opacity 300ms ease; }
.panel-enter-from, .panel-leave-to { transform: translateX(20px); opacity: 0; }

.cursor-blink {
  display: inline-block;
  width: 2px;
  height: 1.1em;
  background: var(--clay);
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: blink 0.8s steps(1) infinite;
}
@keyframes blink { 50% { opacity: 0; } }
</style>
