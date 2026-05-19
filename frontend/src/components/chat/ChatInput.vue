<template>
  <form @submit.prevent="submit" class="flex gap-3">
    <div class="flex-1 relative">
      <input
        ref="inputEl"
        v-model="input"
        type="text"
        placeholder="输入问题，Enter 发送…"
        class="w-full rounded-xl px-4 py-3 text-sm outline-none transition-all duration-200 placeholder:text-[var(--page-faint)]"
        style="background: var(--ink); border: 1px solid var(--border); color: var(--page)"
        :disabled="disabled"
        @keydown.enter="submit"
      />
      <!-- Focus glow -->
      <div class="absolute inset-0 rounded-xl pointer-events-none transition-opacity duration-300"
        :style="{ opacity: focused ? 1 : 0, boxShadow: '0 0 0 2px var(--brass-glow)' }"></div>
    </div>
    <button
      type="submit"
      :disabled="disabled || !input.trim()"
      class="px-6 py-3 rounded-xl text-sm font-medium transition-all duration-200 disabled:opacity-30"
      style="background: var(--brass); color: var(--ink);"
    >
      <span v-if="!disabled">发送</span>
      <span v-else class="inline-block w-4 h-4 border-2 rounded-full animate-spin" style="border-color: var(--ink) transparent var(--ink) transparent"></span>
    </button>
  </form>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ send: [text: string] }>()

const input = ref('')
const focused = ref(false)
const inputEl = ref<HTMLInputElement>()

onMounted(() => inputEl.value?.focus())

function submit() {
  if (!input.value.trim() || props.disabled) return
  emit('send', input.value.trim())
  input.value = ''
}
</script>
