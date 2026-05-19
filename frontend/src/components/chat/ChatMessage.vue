<template>
  <div class="py-2.5" :class="message.role === 'user' ? 'flex justify-end' : 'flex justify-start'">
    <div class="max-w-[72%] rounded-2xl px-5 py-3 text-sm leading-relaxed"
      :class="message.role === 'user' ? 'user-bubble' : 'assistant-bubble'"
    >
      <!-- Role label -->
      <div class="text-[10px] uppercase tracking-widest mb-1.5 opacity-50">
        {{ message.role === 'user' ? 'You' : 'RAG Agent' }}
      </div>

      <div v-html="renderedContent"></div>

      <SourcePanel v-if="message.sources?.length && message.role === 'assistant'"
        :sources="message.sources" :source-type="message.sourceType" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ChatMessage } from '../../types'
import SourcePanel from './SourcePanel.vue'

const props = defineProps<{ message: ChatMessage }>()

const renderedContent = computed(() => {
  return props.message.content.replace(/\n/g, '<br>')
})
</script>

<style scoped>
.user-bubble {
  background: linear-gradient(135deg, var(--brass-dim) 0%, #6b3f1e 100%);
  color: #f0d8b8;
  border-bottom-right-radius: 6px;
}
.assistant-bubble {
  background: var(--ink-muted);
  border: 1px solid var(--border-light);
  color: var(--page);
  border-bottom-left-radius: 6px;
}
</style>
