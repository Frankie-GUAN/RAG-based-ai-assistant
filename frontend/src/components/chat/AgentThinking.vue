<template>
  <div v-if="isStreaming" class="flex items-center gap-3 px-1 py-3">
    <!-- Breathing dot ring -->
    <div class="flex items-center gap-1.5">
      <span class="dot" style="animation-delay: 0ms"></span>
      <span class="dot" style="animation-delay: 150ms"></span>
      <span class="dot" style="animation-delay: 300ms"></span>
    </div>
    <span class="text-xs tracking-wide" style="color: var(--page-dim)">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ route: string; isStreaming: boolean }>()

const label = computed(() => {
  const map: Record<string, string> = {
    'web': 'Searching the web…',
    'rag': 'Retrieving documents…',
    'direct': 'Reasoning…',
  }
  return map[props.route] || 'Thinking…'
})
</script>

<style scoped>
.dot {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--brass);
  animation: breathe 1.2s ease-in-out infinite;
}
@keyframes breathe {
  0%, 60%, 100% { opacity: 0.3; transform: scale(0.8); }
  30% { opacity: 1; transform: scale(1.3); }
}
</style>
