<template>
  <aside class="w-64 bg-white border-r border-gray-100 flex flex-col">
    <div class="p-4 border-b border-gray-100">
      <h1 class="text-xl font-bold text-gray-800">RAG 问答</h1>
    </div>
    <nav class="flex-1 p-4 space-y-2">
      <router-link
        v-for="item in navItems"
        :key="item.path"
        :to="item.path"
        class="block px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        :class="$route.path === item.path ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-50'"
      >
        {{ item.label }}
      </router-link>
    </nav>
  </aside>
</template>

<script setup lang="ts">
const navItems = [
  { path: '/', label: '对话' },
  { path: '/knowledge', label: '知识库管理' },
  { path: '/evaluation', label: 'RAG 评估' },
]
</script>
