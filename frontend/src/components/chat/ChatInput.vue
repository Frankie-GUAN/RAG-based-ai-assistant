<template>
  <form @submit.prevent="submit" class="flex gap-3">
    <input
      v-model="input"
      type="text"
      placeholder="输入你的问题..."
      class="flex-1 rounded-xl border border-gray-200 px-4 py-3 text-sm focus:outline-none focus:border-amber-400 focus:ring-2 focus:ring-amber-100 transition-shadow"
      :disabled="disabled"
    />
    <button
      type="submit"
      :disabled="disabled || !input.trim()"
      class="px-6 py-3 bg-amber-500 text-white rounded-xl text-sm font-medium hover:bg-amber-600 disabled:opacity-40 transition-colors"
    >
      发送
    </button>
  </form>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ send: [text: string] }>()

const input = ref('')

function submit() {
  if (!input.value.trim()) return
  emit('send', input.value.trim())
  input.value = ''
}
</script>
