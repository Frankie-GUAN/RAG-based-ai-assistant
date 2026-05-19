<template>
  <div class="space-y-2">
    <div v-for="doc in documents" :key="doc.id"
      class="flex items-center justify-between px-4 py-3.5 rounded-xl transition-all duration-200"
      style="background: var(--ink-muted); border: 1px solid var(--border)"
    >
      <div class="min-w-0 flex-1">
        <p class="text-sm font-medium truncate" style="color: var(--page)">{{ doc.filename }}</p>
        <p class="text-[11px] mt-0.5 tracking-wide" style="color: var(--page-faint)">
          {{ doc.chunk_count }} chunks · {{ formatDate(doc.created_at) }}
        </p>
      </div>
      <span class="shrink-0 text-[10px] px-2.5 py-1 rounded-full uppercase tracking-wider ml-3"
        style="background: var(--ink); color: var(--page-faint); border: 1px solid var(--border)">
        {{ doc.file_type }}
      </span>
    </div>
    <div v-if="!documents.length" class="text-center py-16" style="color: var(--page-faint)">
      <p class="text-3xl mb-3 opacity-20">⧉</p>
      <p class="text-xs tracking-wide">暂无文档</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DocumentRecord } from '../../types'
defineProps<{ documents: DocumentRecord[] }>()
function formatDate(iso: string) { return new Date(iso).toLocaleDateString('zh-CN') }
</script>
