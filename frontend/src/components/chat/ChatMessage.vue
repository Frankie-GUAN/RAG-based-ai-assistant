<template>
  <div class="py-3" :class="message.role === 'user' ? 'flex justify-end' : 'flex justify-start'">
    <div
      class="max-w-[75%] rounded-xl px-4 py-3 text-sm leading-relaxed"
      :class="message.role === 'user'
        ? 'bg-amber-500 text-white'
        : 'bg-white border border-gray-200 text-gray-800'"
    >
      <div v-html="renderedContent"></div>
      <SourcePanel v-if="message.sources?.length" :sources="message.sources" :source-type="message.sourceType" />
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
