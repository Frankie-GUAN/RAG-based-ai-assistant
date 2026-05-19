<template>
  <div
    class="rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 border-2 border-dashed"
    :style="{
      background: isDragging ? 'var(--clay-wash)' : 'var(--white)',
      borderColor: isDragging ? 'var(--clay)' : 'var(--rule)',
    }"
    @dragover.prevent="isDragging = true"
    @dragleave="isDragging = false"
    @drop.prevent="handleDrop"
    @click="triggerInput"
  >
    <input ref="fileInput" type="file" accept=".pdf,.txt" class="hidden" @change="handleFileChange" />
    <p class="text-5xl mb-4" style="font-family: var(--font-display); color: var(--clay)">+</p>
    <p class="text-sm font-medium">Drop PDF or TXT files here</p>
    <p class="text-xs mt-1.5" style="color: var(--ink-faint)">or click to browse</p>
    <div v-if="isUploading" class="mt-5 flex items-center justify-center gap-2" style="color: var(--clay)">
      <span class="inline-block w-3 h-3 border-2 border-current/30 border-t-current rounded-full animate-spin"></span>
      <span class="text-xs">Processing…</span>
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
  const t = e.target as HTMLInputElement
  if (t.files?.[0]) { emit('upload', t.files[0]); t.value = '' }
}
function handleDrop(e: DragEvent) {
  isDragging.value = false
  if (e.dataTransfer?.files[0]) emit('upload', e.dataTransfer.files[0])
}
</script>
