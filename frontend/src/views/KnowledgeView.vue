<template>
  <div class="flex flex-col h-full">
    <header class="shrink-0 px-6 py-4" style="background: var(--ink-soft); border-bottom: 1px solid var(--border)">
      <h2>知识库管理</h2>
      <p class="text-xs mt-0.5" style="color: var(--page-faint)">上传 PDF 或 TXT 文档，构建私有知识库</p>
    </header>
    <div class="flex-1 overflow-y-auto p-6">
      <FileUploader :is-uploading="isUploading" @upload="handleUpload" />
      <div class="mt-8">
        <h3 class="text-xs uppercase tracking-widest mb-4" style="color: var(--page-faint)">已上传文档</h3>
        <DocList :documents="documents" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useKnowledge } from '../composables/useKnowledge'
import FileUploader from '../components/knowledge/FileUploader.vue'
import DocList from '../components/knowledge/DocList.vue'

const { documents, isUploading, upload, fetchDocuments } = useKnowledge()
onMounted(() => fetchDocuments())

async function handleUpload(file: File) {
  await upload(file)
}
</script>
