<template>
  <div v-if="isStreaming" class="flex items-center gap-2 px-4 py-2 text-xs text-gray-400">
    <span class="inline-block w-2 h-2 bg-amber-400 rounded-full animate-pulse"></span>
    <span>{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ route: string; isStreaming: boolean }>()

const label = computed(() => {
  if (!props.isStreaming) return ''
  const map: Record<string, string> = {
    'web': '正在联网搜索...',
    'rag': '正在检索文档...',
    'direct': '正在生成回答...',
  }
  return map[props.route] || '思考中...'
})
</script>
