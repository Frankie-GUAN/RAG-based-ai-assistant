<template>
  <nav class="flex items-center justify-between h-16 px-8 shrink-0"
    style="background: var(--white-warm); border-bottom: 1px solid var(--rule)">
    <!-- Brand -->
    <router-link to="/" class="flex items-center gap-2.5 no-underline group">
      <span class="text-xl" style="font-family: var(--font-display); color: var(--ink); font-weight: 700">Agentic <span style="color: var(--clay)">RAG</span></span>
      <span class="hidden sm:inline text-[11px] uppercase tracking-widest px-2 py-0.5 rounded-full" style="background: var(--clay-wash); color: var(--clay)">Beta</span>
    </router-link>

    <!-- Nav links -->
    <div class="flex items-center gap-1">
      <router-link v-for="item in navItems" :key="item.path" :to="item.path"
        class="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-200"
        :class="$route.path === item.path
          ? 'active-link'
          : 'text-[var(--ink-muted)] hover:text-[var(--ink)] hover:bg-[var(--rule-faint)]'"
      >
        <span class="text-base">{{ item.icon }}</span>
        <span>{{ item.label }}</span>
      </router-link>
    </div>
  </nav>
</template>

<script setup lang="ts">
const navItems = [
  { path: '/', label: '对话', icon: '◈' },
  { path: '/knowledge', label: '知识库', icon: '⧉' },
  { path: '/evaluation', label: '评估', icon: '⧗' },
]
</script>

<style scoped>
.active-link {
  background: var(--clay-wash);
  color: var(--clay);
}
</style>
