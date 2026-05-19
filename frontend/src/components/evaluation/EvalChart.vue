<template>
  <div class="space-y-3">
    <div v-for="(result, i) in results" :key="i"
      class="rounded-xl p-5 transition-all duration-200"
      style="background: var(--ink-muted); border: 1px solid var(--border)"
    >
      <p class="text-sm font-medium mb-1" style="color: var(--page)">Q: {{ result.query }}</p>
      <p class="text-xs line-clamp-2 mb-4" style="color: var(--page-faint)">A: {{ result.answer }}</p>

      <div class="flex gap-6">
        <!-- Faithfulness -->
        <div class="flex-1">
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-[11px] uppercase tracking-wider" style="color: var(--page-dim)">Faithfulness</span>
            <span class="text-xs font-medium tabular-nums" :style="{ color: result.faithfulness >= 0.7 ? '#7c9a6e' : '#c0655e' }">
              {{ (result.faithfulness * 100).toFixed(0) }}%
            </span>
          </div>
          <div class="h-1.5 rounded-full overflow-hidden" style="background: var(--ink)">
            <div class="h-full rounded-full transition-all duration-700 ease-out"
              :style="{ width: (result.faithfulness * 100) + '%', background: result.faithfulness >= 0.7 ? '#7c9a6e' : '#c0655e' }"></div>
          </div>
        </div>
        <!-- Relevancy -->
        <div class="flex-1">
          <div class="flex items-center justify-between mb-1.5">
            <span class="text-[11px] uppercase tracking-wider" style="color: var(--page-dim)">Relevancy</span>
            <span class="text-xs font-medium tabular-nums" :style="{ color: result.answer_relevancy >= 0.7 ? '#a0875e' : '#c0655e' }">
              {{ (result.answer_relevancy * 100).toFixed(0) }}%
            </span>
          </div>
          <div class="h-1.5 rounded-full overflow-hidden" style="background: var(--ink)">
            <div class="h-full rounded-full transition-all duration-700 ease-out"
              :style="{ width: (result.answer_relevancy * 100) + '%', background: result.answer_relevancy >= 0.7 ? '#a0875e' : '#c0655e' }"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { EvalResult } from '../../types'
defineProps<{ results: EvalResult[] }>()
</script>
