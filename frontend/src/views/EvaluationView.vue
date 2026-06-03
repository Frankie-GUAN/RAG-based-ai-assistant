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
        <div ref="runBtnContainer" class="relative inline-block">
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
          class="result-card rounded-xl p-5" style="background: var(--white); border: 1px solid var(--rule)">
          <p class="text-sm font-semibold mb-1">
            <span class="q-stamp inline-block">Q: </span><span class="q-text inline-block">{{ r.query }}</span>
          </p>
          <p class="text-xs line-clamp-2 mb-4" style="color: var(--ink-muted)">A: {{ r.answer }}</p>
          <div class="result-separator mb-4" style="height: 1px; background: var(--rule); transform-origin: left center;"></div>
          <div class="flex gap-6">
            <div class="flex-1">
              <div class="flex justify-between mb-1">
                <span class="text-[10px] uppercase tracking-wider" style="color: var(--ink-muted)">Faithfulness</span>
                <span class="score-value text-xs font-semibold tabular-nums"
                  :data-score="Math.round(r.faithfulness * 100)"
                  :style="{ color: r.faithfulness >= 0.7 ? 'var(--sage-deep)' : 'var(--clay)' }">0%</span>
              </div>
              <div class="h-1 rounded-full overflow-hidden" style="background: var(--rule-light)">
                <div class="progress-fill h-full rounded-full"
                  :data-target="Math.round(r.faithfulness * 100)"
                  style="width: 0%"
                  :style="{ width: '0%', background: r.faithfulness >= 0.7 ? 'var(--sage)' : 'var(--clay)' }"></div>
              </div>
            </div>
            <div class="flex-1">
              <div class="flex justify-between mb-1">
                <span class="text-[10px] uppercase tracking-wider" style="color: var(--ink-muted)">Relevancy</span>
                <span class="score-value text-xs font-semibold tabular-nums"
                  :data-score="Math.round(r.answer_relevancy * 100)"
                  :style="{ color: r.answer_relevancy >= 0.7 ? 'var(--sage-deep)' : 'var(--clay)' }">0%</span>
              </div>
              <div class="h-1 rounded-full overflow-hidden" style="background: var(--rule-light)">
                <div class="progress-fill h-full rounded-full"
                  :data-target="Math.round(r.answer_relevancy * 100)"
                  style="width: 0%"
                  :style="{ width: '0%', background: r.answer_relevancy >= 0.7 ? 'var(--sage)' : 'var(--clay)' }"></div>
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
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import { useEvaluation } from '../composables/useEvaluation'
import { gsap, initGSAP } from '../composables/useGSAP'
const { results, isRunning, history, run, fetchHistory } = useEvaluation()
const queryText = ref('')
const runBtnContainer = ref<HTMLElement | null>(null)
onMounted(() => {
  initGSAP()
  fetchHistory()
})

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

// 1. Run Button Animation — pulse ring on click
watch(isRunning, (running) => {
  if (running && runBtnContainer.value) {
    const btn = runBtnContainer.value.querySelector('button')
    if (!btn) return
    const ring = document.createElement('div')
    const containerRect = runBtnContainer.value.getBoundingClientRect()
    const btnRect = btn.getBoundingClientRect()
    ring.style.cssText = [
      'position: absolute',
      'border: 2px solid var(--clay)',
      'border-radius: 12px',
      'pointer-events: none',
      `left: ${btnRect.left - containerRect.left}px`,
      `top: ${btnRect.top - containerRect.top}px`,
      `width: ${btnRect.width}px`,
      `height: ${btnRect.height}px`,
    ].join(';')
    runBtnContainer.value.appendChild(ring)
    gsap.fromTo(ring,
      { scale: 1, opacity: 0.8 },
      { scale: 1.5, opacity: 0, duration: 1, ease: 'power3.out', onComplete: () => ring.remove() }
    )
  }
})

// 2. Result Card Print Effect + 3. Progress Bar CountUp
watch(results, async (newResults) => {
  if (!newResults.length) return
  await nextTick()

  const cards = document.querySelectorAll('.result-card')
  cards.forEach((card, i) => {
    const tl = gsap.timeline({ delay: i * 0.1 })

    // Separator expand
    const separator = card.querySelector('.result-separator')
    if (separator) {
      tl.fromTo(separator,
        { scaleX: 0 },
        { scaleX: 1, duration: 0.25, ease: 'power3.out' }
      )
    }

    // Q: stamp
    const stamp = card.querySelector('.q-stamp')
    if (stamp) {
      tl.fromTo(stamp,
        { scale: 0.8, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.2, ease: 'back.out(1.3)' },
        '+=0.05'
      )
    }

    // Query text clip-path reveal
    const qText = card.querySelector('.q-text')
    if (qText) {
      tl.fromTo(qText,
        { clipPath: 'inset(0 100% 0 0)' },
        { clipPath: 'inset(0 0% 0 0)', duration: 0.4, ease: 'power3.inOut' },
        '+=0.03'
      )
    }

    // 3. Progress bars + countUp
    const bars = card.querySelectorAll('.progress-fill')
    bars.forEach((bar) => {
      const targetWidth = parseFloat((bar as HTMLElement).dataset.target || '0')
      const scoreEl = (bar as HTMLElement).closest('.flex-1')?.querySelector('.score-value') as HTMLElement | null
      const targetScore = scoreEl ? parseFloat(scoreEl.dataset.score || '0') : 0

      tl.fromTo(bar,
        { width: '0%' },
        { width: targetWidth + '%', duration: 0.7, ease: 'power3.out' },
        '+=0.05'
      )
      if (scoreEl) {
        const countObj = { val: 0 }
        tl.to(countObj, {
          val: targetScore,
          duration: 0.7,
          ease: 'power2.out',
          onUpdate: () => {
            scoreEl.textContent = Math.round(countObj.val) + '%'
          }
        }, '<0.2')
      }
    })
  })
})
</script>
