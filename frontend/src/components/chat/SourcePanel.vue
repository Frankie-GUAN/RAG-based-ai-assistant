<template>
  <div class="flex flex-wrap gap-2">
    <a v-for="(source, i) in sources" :key="i" :href="source.link" target="_blank"
      class="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full transition-all duration-200 hover:scale-[1.02]"
      style="background: var(--slate-wash); color: var(--slate)">
      <span class="text-[10px]">↗</span>
      <span class="truncate max-w-[180px]">{{ source.title }}</span>
    </a>
  </div>
</template>

<script setup lang="ts">
import type { Source } from '../../types'
defineProps<{ sources: Source[] }>()
</script>
