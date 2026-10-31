<template>
  <div ref="panelRef" class="flex flex-wrap gap-2">
    <template v-for="(source, i) in sources" :key="i">
      <!-- 文档片段没有可点击的 URL：渲染成标签而不是 <a href="">，
           后者看起来可点、点了却什么也不会发生。 -->
      <a v-if="source.link" :href="source.link" target="_blank"
        :ref="el => { if (el) sourceRefs[i] = el as HTMLElement }"
        class="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full transition-all duration-200 hover:scale-[1.02]"
        style="background: var(--slate-wash); color: var(--slate)">
        <span class="text-[10px]">↗</span>
        <span class="truncate max-w-[180px]">{{ source.title }}</span>
      </a>
      <span v-else
        :ref="el => { if (el) sourceRefs[i] = el as HTMLElement }"
        class="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full transition-all duration-200 hover:scale-[1.02]"
        style="background: var(--slate-wash); color: var(--slate)">
        <span class="text-[10px]">§</span>
        <!-- 有章节时文件名压暗、章节突出：用户最关心的是「哪一节」 -->
        <span v-if="source.sourceName" class="truncate max-w-[110px] opacity-60">{{ source.sourceName }}</span>
        <span v-if="source.sourceName && source.section" class="text-[10px] opacity-40">›</span>
        <span class="truncate max-w-[220px]" :title="source.snippet">{{ source.section || source.title }}</span>
      </span>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import type { Source } from '../../types'
import { gsap } from '../../composables/useGSAP'

defineProps<{ sources: Source[] }>()

const panelRef = ref<HTMLElement | null>(null)
const sourceRefs = ref<HTMLElement[]>([])

let ctx: gsap.Context | null = null

onMounted(() => {
  ctx = gsap.context(() => {
    gsap.from(sourceRefs.value.filter(Boolean), {
      opacity: 0,
      y: 6,
      stagger: 0.05,
      duration: 0.3,
      ease: 'power2.out',
    })
  }, panelRef.value ?? undefined)
})

onUnmounted(() => ctx?.revert())
</script>
