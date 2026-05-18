<template>
  <div class="flex flex-col h-full">
    <header class="px-6 py-4 border-b border-gray-100 bg-white">
      <h2 class="text-lg font-semibold text-gray-800">知识库管理</h2>
    </header>
    <div class="flex-1 overflow-y-auto p-6">
      <FileUploader :is-uploading="isUploading" @upload="handleUpload" />
      <div class="mt-6">
        <h3 class="text-sm font-medium text-gray-500 mb-3">已上传文档</h3>
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
