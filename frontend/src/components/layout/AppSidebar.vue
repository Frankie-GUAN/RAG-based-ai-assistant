<template>
  <aside class="w-60 flex flex-col shrink-0" style="background: var(--ink-soft); border-right: 1px solid var(--border)">
    <!-- Brand -->
    <div class="px-5 py-5" style="border-bottom: 1px solid var(--border)">
      <h1 class="text-xl tracking-wide" style="color: var(--page)">
        Agentic <span style="color: var(--brass)">RAG</span>
      </h1>
      <p class="mt-1 text-xs tracking-wider uppercase" style="color: var(--page-faint)">Knowledge Engine</p>
    </div>

    <!-- Nav -->
    <nav class="flex-1 px-3 py-4 space-y-1">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 group"
        :class="$route.path === item.path
          ? 'active-nav'
          : 'text-[var(--page-dim)] hover:text-[var(--page)] hover:bg-[var(--ink-muted)]'"
      >
        <span class="text-base transition-transform duration-200 group-hover:scale-110">
          {{ item.icon }}
        </span>
        <span class="font-medium">{{ item.label }}</span>
        <span v-if="item.shortcut" class="ml-auto text-[10px] opacity-40">{{ item.shortcut }}</span>
      </router-link>
    </nav>

    <!-- Footer -->
    <div class="px-5 py-4" style="border-top: 1px solid var(--border)">
      <div class="flex items-center gap-2">
        <span class="w-1.5 h-1.5 rounded-full animate-pulse" style="background: var(--brass)"></span>
        <span class="text-[11px] tracking-wide" style="color: var(--page-faint)">System Online</span>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
const navItems = [
  { path: '/', label: '对话', icon: '◈', shortcut: '⌘1' },
  { path: '/knowledge', label: '知识库', icon: '⧉', shortcut: '⌘2' },
  { path: '/evaluation', label: '评估', icon: '⧗', shortcut: '⌘3' },
]
</script>

<style scoped>
.active-nav {
  background: var(--brass-wash);
  color: var(--brass) !important;
  box-shadow: inset 2px 0 0 var(--brass);
}
</style>
