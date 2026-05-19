<template>
  <form @submit.prevent="submit" class="relative"
    style="background: var(--white); border: 1px solid var(--rule); border-radius: var(--r-xl); box-shadow: 0 1px 3px rgba(0,0,0,0.03)">
    <textarea
      ref="inputEl"
      v-model="input"
      rows="1"
      placeholder="Ask anything…"
      class="w-full rounded-xl px-5 py-4 text-base outline-none resize-none"
      style="background: transparent; color: var(--ink); min-height: 52px; max-height: 160px; line-height: 1.5"
      :disabled="disabled"
      @keydown.enter.exact.prevent="submit"
      @keydown.enter.shift.exact="input += '\n'"
      @input="autoResize"
      @focus="focused = true"
      @blur="focused = false"
    ></textarea>
    <div class="flex items-center justify-between px-4 pb-3">
      <span class="text-[10px] tracking-wider opacity-0 transition-opacity"
        :style="{ opacity: focused ? 0.4 : 0 }">Enter to send · Shift+Enter for new line</span>
      <button type="submit" :disabled="disabled || !input.trim()"
        class="ml-auto px-4 py-1.5 text-sm font-medium rounded-lg transition-all duration-200 disabled:opacity-30"
        style="background: var(--clay); color: white">
        <span v-if="!disabled">Send</span>
        <span v-else class="flex items-center gap-1.5">
          <span class="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
          Thinking
        </span>
      </button>
    </div>
  </form>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ send: [text: string] }>()

const input = ref('')
const focused = ref(false)
const inputEl = ref<HTMLTextAreaElement>()

onMounted(() => inputEl.value?.focus())

function autoResize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 160) + 'px'
}

function submit() {
  if (!input.value.trim() || props.disabled) return
  emit('send', input.value.trim())
  input.value = ''
  const el = inputEl.value
  if (el) el.style.height = 'auto'
}
</script>
