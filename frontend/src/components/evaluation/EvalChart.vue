<template>
  <div class="space-y-3">
    <div v-for="(result, i) in results" :key="i" class="bg-white border border-gray-100 rounded-lg p-4">
      <p class="text-sm font-medium text-gray-700 mb-1">Q: {{ result.query }}</p>
      <p class="text-xs text-gray-400 mb-3 line-clamp-2">A: {{ result.answer }}</p>
      <div class="flex gap-4">
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-400">忠实度</span>
          <div class="w-20 h-2 bg-gray-100 rounded-full overflow-hidden">
            <div class="h-full bg-emerald-500 rounded-full transition-all" :style="{ width: (result.faithfulness * 100) + '%' }"></div>
          </div>
          <span class="text-xs font-medium" :class="result.faithfulness >= 0.7 ? 'text-emerald-600' : 'text-red-500'">
            {{ (result.faithfulness * 100).toFixed(0) }}%
          </span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-400">相关性</span>
          <div class="w-20 h-2 bg-gray-100 rounded-full overflow-hidden">
            <div class="h-full bg-blue-500 rounded-full transition-all" :style="{ width: (result.answer_relevancy * 100) + '%' }"></div>
          </div>
          <span class="text-xs font-medium" :class="result.answer_relevancy >= 0.7 ? 'text-blue-600' : 'text-red-500'">
            {{ (result.answer_relevancy * 100).toFixed(0) }}%
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { EvalResult } from '../../types'

defineProps<{ results: EvalResult[] }>()
</script>
