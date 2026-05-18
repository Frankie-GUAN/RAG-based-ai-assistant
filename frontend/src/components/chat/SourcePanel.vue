<template>
  <details class="mt-2">
    <summary class="text-xs text-gray-400 cursor-pointer hover:text-gray-500">
      来源链接 ({{ sources.length }} 条)
    </summary>
    <div class="mt-2 space-y-1">
      <a
        v-for="(source, i) in sources"
        :key="i"
        :href="source.link"
        target="_blank"
        class="block text-xs text-amber-600 hover:text-amber-700 truncate"
      >
        {{ source.title }}
      </a>
    </div>
  </details>
</template>

<script setup lang="ts">
import type { Source } from '../../types'

defineProps<{ sources: Source[]; sourceType?: string }>()
</script>
