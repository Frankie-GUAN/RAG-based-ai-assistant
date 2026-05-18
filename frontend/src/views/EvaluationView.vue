<template>
  <div class="flex flex-col h-full">
    <header class="px-6 py-4 border-b border-gray-100 bg-white">
      <h2 class="text-lg font-semibold text-gray-800">RAG 评估</h2>
    </header>
    <div class="flex-1 overflow-y-auto p-6">
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-600 mb-1">测试查询（每行一个）</label>
        <textarea
          v-model="queryText"
          rows="4"
          class="w-full border border-gray-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-amber-400 resize-none"
          placeholder="输入测试查询，每行一个..."
        ></textarea>
        <button
          @click="handleRun"
          :disabled="isRunning || !queryText.trim()"
          class="mt-3 px-5 py-2.5 bg-amber-500 text-white rounded-xl text-sm font-medium hover:bg-amber-600 disabled:opacity-40 transition-colors"
        >
          {{ isRunning ? '评估中...' : '开始评估' }}
        </button>
      </div>

      <EvalChart v-if="results.length" :results="results" />

      <div v-if="history.length && !results.length" class="space-y-2">
        <h3 class="text-sm font-medium text-gray-500 mb-2">历史评估记录</h3>
        <div v-for="h in history" :key="h.id" class="bg-white border border-gray-100 rounded-lg p-3">
          <p class="text-xs text-gray-500 truncate">{{ h.query }}</p>
          <div class="flex gap-4 mt-1">
            <span class="text-xs text-gray-400">忠实度: {{ (h.faithfulness * 100).toFixed(0) }}%</span>
            <span class="text-xs text-gray-400">相关性: {{ (h.answer_relevancy * 100).toFixed(0) }}%</span>
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
