<template>
  <article ref="msgRef" class="py-6" :class="message.role === 'user' ? 'user-msg' : 'assistant-msg'">
    <!-- Horizontal rule -->
    <div ref="ruleRef" class="msg-rule"></div>

    <!-- Label -->
    <div ref="labelRef" class="flex items-center gap-2.5 mb-3">
      <span ref="dotRef" class="label-dot" :class="message.role === 'user' ? 'dot-user' : 'dot-assistant'"></span>
      <span class="text-sm font-semibold uppercase tracking-widest" style="font-family: var(--font-body)"
        :style="{ color: message.role === 'user' ? 'var(--ink-muted)' : 'var(--clay)' }">
        {{ message.role === 'user' ? 'You' : 'RAG Agent' }}
      </span>
      <span class="text-xs tracking-wide opacity-30">{{ message.role === 'assistant' && message.sourceType ? routeLabel : '' }}</span>
    </div>

    <!-- Content -->
    <div ref="contentRef" class="text-lg leading-relaxed" style="color: var(--ink)" v-html="renderedContent"></div>

    <!-- Sources -->
    <div v-if="message.role === 'assistant' && message.sources?.length" class="mt-4">
      <SourcePanel :sources="message.sources" />
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import type { ChatMessage } from '../../types'
import SourcePanel from './SourcePanel.vue'
import { gsap } from '../../composables/useGSAP'

const props = defineProps<{ message: ChatMessage }>()

const msgRef = ref<HTMLElement | null>(null)
const ruleRef = ref<HTMLElement | null>(null)
const dotRef = ref<HTMLElement | null>(null)
const labelRef = ref<HTMLElement | null>(null)
const contentRef = ref<HTMLElement | null>(null)

const renderedContent = computed(() => props.message.content.replace(/\n/g, '<br>'))

const routeLabel = computed(() => {
  const m: Record<string, string> = { web: 'via Web Search', doc: 'via Documents' }
  return m[props.message.sourceType || ''] || ''
})

let ctx: gsap.Context | null = null

onMounted(() => {
  if (!msgRef.value) return
  ctx = gsap.context(() => {
    const tl = gsap.timeline()

    // 1. Rule expands from center
    tl.fromTo(ruleRef.value,
      { scaleX: 0, opacity: 0 },
      { scaleX: 1, opacity: 1, duration: 0.25, ease: 'power3.out' }
    )

    // 2. Label stamp
    tl.fromTo(labelRef.value,
      { scale: 0.8, rotation: props.message.role === 'user' ? 2 : -2, opacity: 0 },
      { scale: 1, rotation: 0, opacity: 1, duration: 0.3, ease: 'back.out(1.3)' },
      '+=0.05'
    )

    // 3. Dot pop (same time as label)
    tl.fromTo(dotRef.value,
      { scale: 0 },
      { scale: 1, duration: 0.2, ease: 'back.out(1.4)' },
      '<'
    )

    // 4. Content clip-path reveal
    tl.fromTo(contentRef.value,
      { clipPath: 'inset(0 100% 0 0)' },
      { clipPath: 'inset(0 0% 0 0)', duration: 0.5, ease: 'power3.inOut' },
      '+=0.05'
    )

    // 5. Subtle shadow
    tl.to(msgRef.value,
      { boxShadow: '0 1px 2px rgba(0,0,0,0.04)', duration: 0.2 },
      '+=0.1'
    )
  }, msgRef.value)
})

onUnmounted(() => {
  ctx?.revert()
})
</script>

<style scoped>
.user-msg {
  padding-left: 0;
  max-width: 92%;
}
.assistant-msg {
  padding-left: 0;
  max-width: 100%;
}

.label-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}
.dot-user { background: var(--ink-subtle); }
.dot-assistant { background: var(--clay); }

.msg-rule {
  height: 1px;
  transform-origin: center;
  background: var(--rule);
  margin-bottom: 1rem;
}
</style>
