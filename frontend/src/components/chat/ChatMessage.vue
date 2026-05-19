<template>
  <article class="py-6" :class="message.role === 'user' ? 'user-msg' : 'assistant-msg'">
    <!-- Label -->
    <div class="flex items-center gap-2.5 mb-3">
      <span class="label-dot" :class="message.role === 'user' ? 'dot-user' : 'dot-assistant'"></span>
      <span class="text-sm font-semibold uppercase tracking-widest" style="font-family: var(--font-body)"
        :style="{ color: message.role === 'user' ? 'var(--ink-muted)' : 'var(--clay)' }">
        {{ message.role === 'user' ? 'You' : 'RAG Agent' }}
      </span>
      <span class="text-xs tracking-wide opacity-30">{{ message.role === 'assistant' && message.sourceType ? routeLabel : '' }}</span>
    </div>

    <!-- Content -->
    <div class="text-lg leading-relaxed" style="color: var(--ink)" v-html="renderedContent"></div>

    <!-- Sources -->
    <div v-if="message.role === 'assistant' && message.sources?.length" class="mt-4">
      <SourcePanel :sources="message.sources" />
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ChatMessage } from '../../types'
import SourcePanel from './SourcePanel.vue'

const props = defineProps<{ message: ChatMessage }>()

const renderedContent = computed(() => props.message.content.replace(/\n/g, '<br>'))

const routeLabel = computed(() => {
  const m: Record<string, string> = { web: 'via Web Search', doc: 'via Documents' }
  return m[props.message.sourceType || ''] || ''
})
</script>

<style scoped>
.user-msg {
  padding-left: 0;
  max-width: 92%;
}
.assistant-msg {
  padding-left: 0;
  max-width: 100%;
}

.label-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}
.dot-user { background: var(--ink-subtle); }
.dot-assistant { background: var(--clay); }
</style>
