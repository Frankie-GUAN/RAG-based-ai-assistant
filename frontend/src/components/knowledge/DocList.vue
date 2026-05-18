<template>
  <div class="space-y-2">
    <div
      v-for="doc in documents"
      :key="doc.id"
      class="flex items-center justify-between px-4 py-3 bg-white border border-gray-100 rounded-lg"
    >
      <div>
        <p class="text-sm font-medium text-gray-700">{{ doc.filename }}</p>
        <p class="text-xs text-gray-400">{{ doc.chunk_count }} 个片段 · {{ formatDate(doc.created_at) }}</p>
      </div>
      <span class="text-xs px-2 py-1 bg-gray-100 rounded text-gray-500">{{ doc.file_type.toUpperCase() }}</span>
    </div>
    <div v-if="!documents.length" class="text-center py-12 text-sm text-gray-400">
      暂无文档，请上传 PDF 或 TXT 文件
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DocumentRecord } from '../../types'

defineProps<{ documents: DocumentRecord[] }>()

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('zh-CN')
}
</script>
