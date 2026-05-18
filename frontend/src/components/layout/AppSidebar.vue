<template>
  <aside class="w-56 bg-white border-r border-gray-200 flex flex-col">
    <div class="p-4 border-b border-gray-100">
      <h1 class="text-lg font-semibold text-gray-800">Agentic RAG</h1>
      <p class="text-xs text-gray-400 mt-0.5">智能问答平台</p>
    </div>
    <nav class="flex-1 p-3 space-y-1">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        class="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-gray-600 hover:bg-gray-100 transition-colors"
        active-class="bg-amber-50 text-amber-700 font-medium"
      >
        <span>{{ item.icon }}</span>
        <span>{{ item.label }}</span>
      </router-link>
    </nav>
    <div class="p-3 border-t border-gray-100">
      <div class="text-xs text-gray-400">Agentic RAG v0.1</div>
    </div>
  </aside>
</template>

<script setup lang="ts">
const navItems = [
  { path: '/', label: '对话', icon: '💬' },
  { path: '/knowledge', label: '知识库', icon: '📚' },
  { path: '/evaluation', label: '评估', icon: '📊' },
]
</script>
