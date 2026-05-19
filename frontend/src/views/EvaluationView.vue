<template>
  <div class="h-full overflow-y-auto">
    <div class="max-w-4xl mx-auto px-10 py-14">
      <h1 class="text-4xl mb-3" style="font-family: var(--font-display)">Evaluation</h1>
      <p class="text-lg mb-12 max-w-2xl" style="color: var(--ink-muted)">Measure retrieval quality. Each query runs through the full Agent pipeline and is scored on faithfulness and relevancy.</p>

      <!-- Query input card -->
      <div class="rounded-2xl p-6 mb-10" style="background: var(--white); border: 1px solid var(--rule)">
        <label class="block text-xs uppercase tracking-widest mb-3 font-semibold" style="color: var(--ink-muted); font-family: var(--font-body)">Test Queries</label>
        <textarea v-model="queryText" rows="5"
          class="w-full rounded-xl px-4 py-3 text-sm outline-none resize-none font-mono"
          style="background: var(--paper); border: 1px solid var(--rule); color: var(--ink); line-height: 1.7"
          placeholder="One query per line&#10;&#10;Python 是什么？&#10;如何解除劳动合同？&#10;2025 年劳动法有哪些更新？"></textarea>
        <button @click="handleRun" :disabled="isRunning || !queryText.trim()"
          class="mt-4 px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 disabled:opacity-30"
          style="background: var(--clay); color: white">
          <span v-if="!isRunning">Run Evaluation</span>
          <span v-else class="flex items-center gap-2">
            <span class="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
            Evaluating…
          </span>
        </button>
      </div>

      <!-- Results -->
      <div v-if="results.length" class="space-y-4">
        <h2 class="text-xs uppercase tracking-widest mb-4 font-semibold" style="color: var(--ink-muted); font-family: var(--font-body)">Results</h2>

        <!-- Metric cards — horizontal -->
        <div class="grid grid-cols-2 gap-4 mb-6">
          <div class="rounded-xl p-5" style="background: var(--white); border: 1px solid var(--rule)">
            <p class="text-[10px] uppercase tracking-widest mb-1" style="color: var(--ink-muted)">Avg Faithfulness</p>
            <p class="text-3xl" style="font-family: var(--font-display); color: var(--sage)">{{ avgFaithfulness }}%</p>
          </div>
          <div class="rounded-xl p-5" style="background: var(--white); border: 1px solid var(--rule)">
            <p class="text-[10px] uppercase tracking-widest mb-1" style="color: var(--ink-muted)">Avg Relevancy</p>
            <p class="text-3xl" style="font-family: var(--font-display); color: var(--clay)">{{ avgRelevancy }}%</p>
          </div>
        </div>

        <!-- Individual results -->
        <div v-for="(r, i) in results" :key="i"
          class="rounded-xl p-5" style="background: var(--white); border: 1px solid var(--rule)">
          <p class="text-sm font-semibold mb-1">Q: {{ r.query }}</p>
          <p class="text-xs line-clamp-2 mb-4" style="color: var(--ink-muted)">A: {{ r.answer }}</p>
          <div class="flex gap-6">
            <div class="flex-1">
              <div class="flex justify-between mb-1"><span class="text-[10px] uppercase tracking-wider" style="color: var(--ink-muted)">Faithfulness</span><span class="text-xs font-semibold tabular-nums" :style="{ color: r.faithfulness >= 0.7 ? 'var(--sage-deep)' : 'var(--clay)' }">{{ (r.faithfulness * 100).toFixed(0) }}%</span></div>
              <div class="h-1 rounded-full overflow-hidden" style="background: var(--rule-light)">
                <div class="h-full rounded-full transition-all duration-700" :style="{ width: (r.faithfulness * 100) + '%', background: r.faithfulness >= 0.7 ? 'var(--sage)' : 'var(--clay)' }"></div>
              </div>
            </div>
            <div class="flex-1">
              <div class="flex justify-between mb-1"><span class="text-[10px] uppercase tracking-wider" style="color: var(--ink-muted)">Relevancy</span><span class="text-xs font-semibold tabular-nums" :style="{ color: r.answer_relevancy >= 0.7 ? 'var(--sage-deep)' : 'var(--clay)' }">{{ (r.answer_relevancy * 100).toFixed(0) }}%</span></div>
              <div class="h-1 rounded-full overflow-hidden" style="background: var(--rule-light)">
                <div class="h-full rounded-full transition-all duration-700" :style="{ width: (r.answer_relevancy * 100) + '%', background: r.answer_relevancy >= 0.7 ? 'var(--sage)' : 'var(--clay)' }"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- History -->
      <div v-if="history.length && !results.length" class="mt-6">
        <h2 class="text-xs uppercase tracking-widest mb-4 font-semibold" style="color: var(--ink-muted); font-family: var(--font-body)">History</h2>
        <div class="space-y-2">
          <div v-for="h in history" :key="h.id" class="flex items-center justify-between px-4 py-3 rounded-xl" style="background: var(--white); border: 1px solid var(--rule)">
            <p class="text-xs truncate flex-1 mr-4" style="color: var(--ink-muted)">{{ h.query }}</p>
            <div class="flex gap-4 text-xs font-semibold shrink-0">
              <span :style="{ color: h.faithfulness >= 0.7 ? 'var(--sage)' : 'var(--clay)' }">F {{ (h.faithfulness * 100).toFixed(0) }}%</span>
              <span :style="{ color: h.answer_relevancy >= 0.7 ? 'var(--sage)' : 'var(--clay)' }">R {{ (h.answer_relevancy * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useEvaluation } from '../composables/useEvaluation'
const { results, isRunning, history, run, fetchHistory } = useEvaluation()
const queryText = ref('')
onMounted(() => fetchHistory())

const avgFaithfulness = computed(() => {
  if (!results.value.length) return 0
  const sum = results.value.reduce((a, r) => a + (r.faithfulness || 0), 0)
  return Math.round((sum / results.value.length) * 100)
})
const avgRelevancy = computed(() => {
  if (!results.value.length) return 0
  const sum = results.value.reduce((a, r) => a + (r.answer_relevancy || 0), 0)
  return Math.round((sum / results.value.length) * 100)
})

async function handleRun() {
  const queries = queryText.value.split('\n').filter(q => q.trim())
  if (!queries.length) return
  await run(queries, false)
}
</script>
