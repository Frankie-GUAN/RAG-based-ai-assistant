<template>
  <details class="mt-2.5">
    <summary class="text-[11px] cursor-pointer opacity-40 hover:opacity-70 transition-opacity tracking-wide uppercase">
      {{ sources.length }} Source{{ sources.length > 1 ? 's' : '' }}
    </summary>
    <div class="mt-2 space-y-1.5 pl-2" style="border-left: 1px solid var(--border)">
      <a v-for="(source, i) in sources" :key="i"
        :href="source.link" target="_blank"
        class="block text-xs truncate opacity-60 hover:opacity-100 transition-opacity"
        style="color: var(--brass)">
        {{ source.title }}
      </a>
    </div>
  </details>
</template>

<script setup lang="ts">
import type { Source } from '../../types'
defineProps<{ sources: Source[]; sourceType?: string }>()
</script>
