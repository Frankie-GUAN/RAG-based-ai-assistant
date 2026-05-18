<template>
  <div
    class="border-2 border-dashed border-gray-200 rounded-xl p-8 text-center hover:border-amber-300 hover:bg-amber-50/50 transition-colors cursor-pointer"
    @dragover.prevent
    @drop.prevent="handleDrop"
    @click="triggerInput"
  >
    <input ref="fileInput" type="file" accept=".pdf,.txt" class="hidden" @change="handleFileChange" />
    <div class="text-3xl mb-2">📄</div>
    <p class="text-sm text-gray-500">拖拽 PDF 或 TXT 文件到此处，或点击上传</p>
    <p v-if="isUploading" class="text-xs text-amber-500 mt-2">上传中...</p>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ isUploading: boolean }>()
const emit = defineEmits<{ upload: [file: File] }>()

const fileInput = ref<HTMLInputElement>()

function triggerInput() {
  fileInput.value?.click()
}

function handleFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files?.[0]) {
    emit('upload', target.files[0])
    target.value = ''
  }
}

function handleDrop(e: DragEvent) {
  if (e.dataTransfer?.files[0]) {
    emit('upload', e.dataTransfer.files[0])
  }
}
</script>
