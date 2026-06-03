<template>
  <div v-if="isStreaming" class="flex items-center gap-3 py-4">
    <div class="flex items-center gap-1">
      <span ref="dot1" class="think-dot"></span>
      <span ref="dot2" class="think-dot"></span>
      <span ref="dot3" class="think-dot"></span>
    </div>
    <span ref="labelEl" class="text-xs tracking-wide italic" style="color: var(--ink-muted)">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed, watch, nextTick, onUnmounted, ref } from 'vue'
import { gsap } from '../../composables/useGSAP'

const props = defineProps<{ route: string; isStreaming: boolean }>()
const label = computed(() => {
  const m: Record<string, string> = {
    web: 'Searching the web…',
    rag: 'Reading your documents…',
    direct: 'Reasoning…',
  }
  return m[props.route] || 'Thinking…'
})

const dot1 = ref<HTMLElement | null>(null)
const dot2 = ref<HTMLElement | null>(null)
const dot3 = ref<HTMLElement | null>(null)
const labelEl = ref<HTMLElement | null>(null)

let ctx: gsap.Context | null = null

function resolveCSSVar(name: string, fallback: string): string {
  const val = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return val || fallback
}

function setupAnimation() {
  ctx?.revert()
  if (!props.isStreaming) return

  const inkSubtle = resolveCSSVar('--ink-subtle', '#8B8B8B')
  const clay = resolveCSSVar('--clay', '#C4A882')

  ctx = gsap.context(() => {
    const dots = [dot1.value, dot2.value, dot3.value].filter(Boolean) as HTMLElement[]

    // Wave dot timeline — each dot pulses staggered 0.15s apart
    const tl = gsap.timeline({ repeat: -1 })
    dots.forEach((dot, i) => {
      tl.fromTo(
        dot,
        { scale: 0.4 },
        {
          scale: 1,
          opacity: 1,
          backgroundColor: clay,
          duration: 0.45,
          ease: 'sine.inOut',
        },
        i * 0.15,
      )
      tl.to(dot, {
        scale: 0.4,
        opacity: 0.2,
        backgroundColor: inkSubtle,
        duration: 0.45,
        ease: 'sine.inOut',
      })
    })

    // Label fade pulse
    gsap.fromTo(
      labelEl.value,
      { opacity: 0.6 },
      {
        opacity: 1,
        duration: 0.9,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut',
      },
    )
  })
}

watch(
  () => props.isStreaming,
  async (streaming) => {
    if (streaming) {
      await nextTick()
      setupAnimation()
    } else {
      ctx?.revert()
      ctx = null
    }
  },
  { immediate: true },
)

onUnmounted(() => {
  ctx?.revert()
})
</script>

<style scoped>
.think-dot {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: var(--ink-subtle);
  opacity: 0.2;
}
</style>
