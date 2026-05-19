<template>
  <div
    class="rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 border-2 border-dashed"
    :style="{
      background: isDragging ? 'var(--brass-wash)' : 'var(--ink-muted)',
      borderColor: isDragging ? 'var(--brass)' : 'var(--border)',
    }"
    @dragover.prevent="isDragging = true"
    @dragleave="isDragging = false"
    @drop.prevent="handleDrop"
    @click="triggerInput"
  >
    <input ref="fileInput" type="file" accept=".pdf,.txt" class="hidden" @change="handleFileChange" />
    <div class="text-4xl mb-3 opacity-50">⧉</div>
    <p class="text-sm" style="color: var(--page-dim)">拖拽 PDF 或 TXT 到此处，或点击上传</p>
    <p class="text-[11px] mt-2 tracking-wide" style="color: var(--page-faint)">支持 .pdf / .txt 格式</p>
    <div v-if="isUploading" class="mt-4 flex items-center justify-center gap-2" style="color: var(--brass)">
      <span class="inline-block w-3 h-3 border-2 rounded-full animate-spin" style="border-color: var(--brass) transparent var(--brass) transparent"></span>
      <span class="text-xs">处理中…</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{ isUploading: boolean }>()
const emit = defineEmits<{ upload: [file: File] }>()

const fileInput = ref<HTMLInputElement>()
const isDragging = ref(false)

function triggerInput() { fileInput.value?.click() }
function handleFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  if (target.files?.[0]) { emit('upload', target.files[0]); target.value = '' }
}
function handleDrop(e: DragEvent) {
  isDragging.value = false
  if (e.dataTransfer?.files[0]) emit('upload', e.dataTransfer.files[0])
}
</script>
