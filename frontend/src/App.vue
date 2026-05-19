<template>
  <div class="flex flex-col h-screen" style="background: var(--paper)">
    <TopNav />
    <main class="flex-1 overflow-hidden">
      <router-view v-slot="{ Component }">
        <transition name="page" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import TopNav from './components/layout/TopNav.vue'
</script>

<style>
.page-enter-active,
.page-leave-active {
  transition: opacity 240ms ease, transform 240ms ease;
}
.page-enter-from { opacity: 0; transform: translateY(6px) scale(0.995); }
.page-leave-to   { opacity: 0; transform: translateY(-4px) scale(0.995); }
</style>
