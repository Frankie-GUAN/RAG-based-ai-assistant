<template>
  <nav ref="navRef" class="flex items-center justify-between h-16 px-8 shrink-0"
    style="background: var(--white-warm); border-bottom: 1px solid var(--rule)">
    <!-- Brand -->
    <router-link to="/" class="flex items-center gap-2.5 no-underline group">
      <span class="text-xl" style="font-family: var(--font-display); color: var(--ink); font-weight: 700">Agentic <span style="color: var(--clay)">RAG</span></span>
      <span class="hidden sm:inline text-[11px] uppercase tracking-widest px-2 py-0.5 rounded-full" style="background: var(--clay-wash); color: var(--clay)">Beta</span>
    </router-link>

    <!-- Nav links -->
    <div class="flex items-center gap-1">
      <router-link v-for="item in navItems" :key="item.path" :to="item.path"
        class="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200 relative"
        :class="$route.path === item.path
          ? 'active-link'
          : 'text-[var(--ink-muted)] hover:text-[var(--ink)] hover:bg-[var(--rule-faint)]'"
      >
        <span class="text-base">{{ item.icon }}</span>
        <span>{{ item.label }}</span>
        <span
          v-if="$route.path === item.path"
          class="active-indicator absolute bottom-0 left-2 right-2 h-0.5 rounded-full"
          style="background: var(--clay); transform-origin: left;"
        />
      </router-link>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { gsap } from '../../composables/useGSAP'

const route = useRoute()
let ctx: gsap.Context | null = null

const navRef = ref<HTMLElement | null>(null)
const navItems: Array<{ path: string; label: string; icon: string }> = [
  { path: '/', label: '对话', icon: '◈' },
  { path: '/knowledge', label: '知识库', icon: '⧉' },
  { path: '/evaluation', label: '评估', icon: '⧗' },
]

onMounted(() => {
  const el = navRef.value
  if (!el) return
  ctx = gsap.context(() => {
    gsap.from(el, {
      y: -48,
      duration: 0.5,
      ease: 'power3.out',
      delay: 0.3,
    })
  }, el)
})

onUnmounted(() => {
  ctx?.revert()
})

watch(() => route.path, async () => {
  await nextTick()
  const indicator = navRef.value?.querySelector('.active-link .active-indicator') as HTMLElement | null
  if (indicator) {
    gsap.fromTo(indicator,
      { scaleX: 0.6, opacity: 0 },
      { scaleX: 1, opacity: 1, duration: 0.3, ease: 'power2.inOut' }
    )
  }
})
</script>

<style scoped>
.active-link {
  background: var(--clay-wash);
  color: var(--clay);
}
</style>
