<template>
  <div
    ref="uploaderRef"
    class="rounded-2xl p-16 text-center cursor-pointer transition-all duration-300 border-2 border-dashed"
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
    <div v-if="!filename" class="flex flex-col items-center">
      <p class="text-5xl mb-4" style="font-family: var(--font-display); color: var(--clay)">+</p>
      <p class="text-sm font-medium">Drop PDF or TXT files here</p>
      <p class="text-xs mt-1.5" style="color: var(--ink-faint)">or click to browse</p>
    </div>
    <p
      v-else
      ref="filenameLabel"
      class="text-sm font-medium"
      style="color: var(--clay)"
    >{{ filename }}</p>
    <div v-if="isUploading" class="mt-5 flex items-center justify-center gap-2" style="color: var(--clay)">
      <span class="inline-block w-3 h-3 border-2 border-current/30 border-t-current rounded-full animate-spin"></span>
      <span class="text-xs">Processing…</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'
import gsap from '../../composables/useGSAP'
defineProps<{ isUploading: boolean }>()
const emit = defineEmits<{ upload: [file: File] }>()
const fileInput = ref<HTMLInputElement>()
const uploaderRef = ref<HTMLDivElement>()
const filenameLabel = ref<HTMLParagraphElement>()
const isDragging = ref(false)
const filename = ref('')

function playBreatheAnimation() {
  const el = uploaderRef.value
  if (!el) return
  const tl = gsap.timeline({
    onComplete: () => {
      gsap.set(el, { clearProps: 'borderColor,boxShadow' })
    },
  })
  // Border breathe: rule -> clay -> rule with boxShadow pulse
  tl.to(el, {
    borderColor: 'var(--clay)',
    boxShadow: '0 0 0 8px var(--clay-glow)',
    duration: 0.3,
    ease: 'power1.out',
  }).to(el, {
    borderColor: 'var(--rule)',
    boxShadow: '0 0 0 0px var(--clay-glow)',
    duration: 0.3,
    ease: 'power1.in',
  })
  // Filename label pop-in, starting 0.1s after breathe begins
  if (filenameLabel.value) {
    tl.fromTo(
      filenameLabel.value,
      { opacity: 0, y: -12, scale: 0.9 },
      {
        opacity: 1,
        y: 0,
        scale: 1,
        duration: 0.35,
        ease: 'back.out(1.3)',
      },
      0.1
    )
  }
}

function triggerInput() { fileInput.value?.click() }
function handleFileChange(e: Event) {
  const t = e.target as HTMLInputElement
  if (t.files?.[0]) {
    const file = t.files[0]
    filename.value = file.name
    emit('upload', file)
    t.value = ''
    nextTick(() => playBreatheAnimation())
  }
}
function handleDrop(e: DragEvent) {
  isDragging.value = false
  const file = e.dataTransfer?.files[0]
  if (file) {
    filename.value = file.name
    emit('upload', file)
    nextTick(() => playBreatheAnimation())
  }
}
</script>
