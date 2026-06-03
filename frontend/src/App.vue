<template>
  <div class="flex flex-col h-screen" style="background: var(--paper)">
    <TopNav />
    <main class="flex-1 overflow-hidden">
      <router-view v-slot="{ Component }">
        <transition mode="out-in" :css="false" @leave="onLeave" @enter="onEnter">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import TopNav from './components/layout/TopNav.vue'
import { gsap } from './composables/useGSAP'

function onLeave(el: Element, done: () => void) {
  gsap.to(el, {
    opacity: 0,
    y: -8,
    scale: 0.995,
    duration: 0.2,
    ease: 'power2.in',
    onComplete: done,
  })
}

function onEnter(el: Element, done: () => void) {
  gsap.fromTo(
    el,
    { opacity: 0, y: 12 },
    { opacity: 1, y: 0, duration: 0.35, ease: 'power3.out', onComplete: done },
  )
}
</script>

