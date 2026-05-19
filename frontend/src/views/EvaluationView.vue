<template>
  <div class="flex flex-col h-full">
    <header class="shrink-0 px-6 py-4" style="background: var(--ink-soft); border-bottom: 1px solid var(--border)">
      <h2>RAG 评估</h2>
      <p class="text-xs mt-0.5" style="color: var(--page-faint)">运行测试查询，评估检索与生成质量</p>
    </header>
    <div class="flex-1 overflow-y-auto p-6">
      <!-- Input area -->
      <div class="rounded-2xl p-5 mb-8" style="background: var(--ink-muted); border: 1px solid var(--border)">
        <label class="block text-xs uppercase tracking-widest mb-3" style="color: var(--page-dim)">测试查询</label>
        <textarea v-model="queryText" rows="4"
          class="w-full rounded-xl px-4 py-3 text-sm outline-none resize-none transition-all duration-200 placeholder:text-[var(--page-faint)]"
          style="background: var(--ink); border: 1px solid var(--border); color: var(--page)"
          placeholder="每行一个测试查询&#10;例如：&#10;Python 是什么？&#10;劳动法关于解除合同的规定？"></textarea>
        <button @click="handleRun"
          :disabled="isRunning || !queryText.trim()"
          class="mt-4 px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 disabled:opacity-30"
          style="background: var(--brass); color: var(--ink)">
          <span v-if="!isRunning">开始评估</span>
          <span v-else class="flex items-center gap-2">
            <span class="inline-block w-3 h-3 border-2 rounded-full animate-spin" style="border-color: var(--ink) transparent var(--ink) transparent"></span>
            评估中…
          </span>
        </button>
      </div>

      <!-- Results -->
      <EvalChart v-if="results.length" :results="results" />

      <!-- History -->
      <div v-if="history.length && !results.length">
        <h3 class="text-xs uppercase tracking-widest mb-4" style="color: var(--page-faint)">历史记录</h3>
        <div class="space-y-2">
          <div v-for="h in history" :key="h.id"
            class="rounded-xl p-4 transition-all duration-200"
            style="background: var(--ink-muted); border: 1px solid var(--border)">
            <p class="text-xs truncate opacity-50">{{ h.query }}</p>
            <div class="flex gap-6 mt-2 text-xs tracking-wide" style="color: var(--page-dim)">
              <span>忠实度 {{ (h.faithfulness * 100).toFixed(0) }}%</span>
              <span>相关性 {{ (h.answer_relevancy * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useEvaluation } from '../composables/useEvaluation'
import EvalChart from '../components/evaluation/EvalChart.vue'

const { results, isRunning, history, run, fetchHistory } = useEvaluation()
const queryText = ref('')
onMounted(() => fetchHistory())

async function handleRun() {
  const queries = queryText.value.split('\n').filter(q => q.trim())
  if (!queries.length) return
  await run(queries, false)
}
</script>
