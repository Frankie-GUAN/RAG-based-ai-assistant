import { ref } from 'vue'
import type { EvalResult } from '../types'

export function useEvaluation() {
  const results = ref<EvalResult[]>([])
  const isRunning = ref(false)
  const history = ref<any[]>([])

  async function run(queries: string[], hasDocs: boolean) {
    isRunning.value = true
    const response = await fetch('/api/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ queries, has_docs: hasDocs }),
    })
    const data = await response.json()
    results.value = data.results
    isRunning.value = false
  }

  async function fetchHistory() {
    const response = await fetch('/api/evaluation/history')
    history.value = await response.json()
  }

  return { results, isRunning, history, run, fetchHistory }
}
