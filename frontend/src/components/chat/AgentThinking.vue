<template>
  <div v-if="isStreaming" class="flex items-center gap-3 py-4">
    <div class="flex items-center gap-1">
      <span class="think-dot" style="animation-delay: 0ms"></span>
      <span class="think-dot" style="animation-delay: 180ms"></span>
      <span class="think-dot" style="animation-delay: 360ms"></span>
    </div>
    <span class="text-xs tracking-wide italic" style="color: var(--ink-muted)">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
const props = defineProps<{ route: string; isStreaming: boolean }>()
const label = computed(() => {
  const m: Record<string, string> = {
    'web': 'Searching the web…',
    'rag': 'Reading your documents…',
    'direct': 'Reasoning…',
  }
  return m[props.route] || 'Thinking…'
})
</script>

<style scoped>
.think-dot { width: 3px; height: 3px; border-radius: 50%; background: var(--clay); animation: think 1.4s ease-in-out infinite; }
@keyframes think { 0%, 60%, 100% { opacity: 0.2; transform: scale(0.6); } 30% { opacity: 1; transform: scale(1); } }
</style>
