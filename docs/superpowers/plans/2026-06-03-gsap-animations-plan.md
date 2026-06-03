# GSAP Frontend Animations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rich, cohesive GSAP animation language to the RAG chatbot frontend following the "Print Impression" metaphor.

**Architecture:** Create a shared `useGSAP` composable for GSAP instance + plugin registration + `gsap.matchMedia()` setup. Each component uses `<script setup>` with `onMounted`/`onUnmounted` and `gsap.context(scope)` for scoped selectors and auto-cleanup. ScrollTrigger instances live in the same contexts and are auto-killed on unmount.

**Tech Stack:** Vue 3 (Composition API, `<script setup>`) + TypeScript + Tailwind CSS v4 + GSAP 3 + ScrollTrigger plugin

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/composables/useGSAP.ts` | Shared GSAP instance, plugin registration, matchMedia setup |
| Modify | `src/main.ts` | Call `initGSAP()` on app startup |
| Modify | `src/App.vue` | GSAP route transition (replace CSS `<transition>`) |
| Modify | `src/components/layout/TopNav.vue` | Nav slide-in, active indicator slide |
| Modify | `src/components/chat/ChatMessage.vue` | Print impression message enter |
| Modify | `src/components/chat/AgentThinking.vue` | Wave dot loop + label pulse |
| Modify | `src/components/chat/ChatInput.vue` | Send button feedback animation |
| Modify | `src/components/chat/SourcePanel.vue` | Source item stagger reveal |
| Modify | `src/components/chat/ConversationSidebar.vue` | Panel enter/exit, item hover, delete animation |
| Modify | `src/views/ChatView.vue` | Empty state animation, smooth scroll, streaming cursor, context panel |
| Modify | `src/components/knowledge/FileUploader.vue` | File drop border breathe + filename pop-in |
| Modify | `src/views/KnowledgeView.vue` | Card enter + ScrollTrigger reveal |
| Modify | `src/views/EvaluationView.vue` | Result card reveal, progress bar animation, countUp |

---

### Task 1: Install GSAP and create useGSAP composable

**Files:**
- Modify: `frontend/src/composables/useGSAP.ts` (create)
- Modify: `frontend/src/main.ts`

- [ ] **Step 1: Install gsap npm package**

```bash
cd frontend && npm install gsap
```
Expected: `gsap` added to `package.json` dependencies.

- [ ] **Step 2: Create shared useGSAP composable**

Write `frontend/src/composables/useGSAP.ts`:

```typescript
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

let initialized = false

export function initGSAP() {
  if (initialized) return
  gsap.registerPlugin(ScrollTrigger)

  // Global reduced-motion support
  const mm = gsap.matchMedia()
  mm.add('(prefers-reduced-motion: reduce)', () => {
    gsap.defaults({ duration: 0, overwrite: true })
  })

  initialized = true
}

export { gsap, ScrollTrigger }
```

- [ ] **Step 3: Wire initGSAP into main.ts**

Modify `frontend/src/main.ts` — add import and call after app creation but before mount:

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './styles/main.css'
import { initGSAP } from './composables/useGSAP'

const app = createApp(App)
app.use(createPinia())
app.use(router)

initGSAP()  // register ScrollTrigger + matchMedia

app.mount('#app')
```

- [ ] **Step 4: Commit**

```bash
cd frontend && npm install gsap
git add frontend/package.json frontend/package-lock.json frontend/src/composables/useGSAP.ts frontend/src/main.ts
git commit -m "chore: add GSAP dependency and useGSAP composable

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: GSAP route transition in App.vue

**Files:**
- Modify: `frontend/src/App.vue`

- [ ] **Step 1: Replace CSS transition with GSAP route transition**

Modify `frontend/src/App.vue`:

```vue
<template>
  <div class="flex flex-col h-screen" style="background: var(--paper)">
    <TopNav />
    <main class="flex-1 overflow-hidden" ref="mainRef">
      <router-view v-slot="{ Component, route }">
        <div :key="route.path" ref="pageRef">
          <component :is="Component" />
        </div>
      </router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import TopNav from './components/layout/TopNav.vue'
import { gsap } from './composables/useGSAP'

const router = useRouter()
const mainRef = ref<HTMLElement>()
const pageRef = ref<HTMLElement>()
let ctx: gsap.Context | null = null

onMounted(() => {
  if (!mainRef.value) return
  ctx = gsap.context(() => {}, mainRef.value)
})

onUnmounted(() => {
  ctx?.revert()
})

// Watch route changes for transition
router.beforeEach((_to, _from, next) => {
  if (pageRef.value && ctx) {
    // Exit current page
    gsap.to(pageRef.value, {
      opacity: 0,
      y: -8,
      scale: 0.995,
      duration: 0.2,
      ease: 'power2.in',
      onComplete: () => next(),
    })
  } else {
    next()
  }
})

// After route change, animate new page in
watch(() => router.currentRoute.value, () => {
  // Wait for DOM update then animate in
  setTimeout(() => {
    if (!pageRef.value) return
    gsap.fromTo(pageRef.value,
      { opacity: 0, y: 12 },
      { opacity: 1, y: 0, duration: 0.35, ease: 'power3.out' }
    )
  }, 50)
})
</script>

<style>
/* Remove old CSS transition styles — delete .page-enter-active, .page-leave-active, etc. */
</style>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/App.vue
git commit -m "feat: replace CSS route transition with GSAP animation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: TopNav animations

**Files:**
- Modify: `frontend/src/components/layout/TopNav.vue`

- [ ] **Step 1: Add nav slide-in and active indicator animation**

Replace the existing `<script setup>` block and add GSAP animations:

```vue
<template>
  <nav ref="navRef" class="flex items-center justify-between h-16 px-8 shrink-0"
    style="background: var(--white-warm); border-bottom: 1px solid var(--rule)">
    <!-- Brand -->
    <router-link to="/" class="flex items-center gap-2.5 no-underline group">
      <span class="text-xl" style="font-family: var(--font-display); color: var(--ink); font-weight: 700">
        Agentic <span style="color: var(--clay)">RAG</span>
      </span>
      <span class="hidden sm:inline text-[11px] uppercase tracking-widest px-2 py-0.5 rounded-full"
        style="background: var(--clay-wash); color: var(--clay)">Beta</span>
    </router-link>

    <!-- Nav links -->
    <div ref="navLinksRef" class="flex items-center gap-1">
      <router-link
        v-for="item in navItems" :key="item.path" :to="item.path"
        ref="linkRefs"
        class="nav-link flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors duration-200 relative"
        :class="$route.path === item.path
          ? 'active-link'
          : 'text-[var(--ink-muted)] hover:text-[var(--ink)] hover:bg-[var(--rule-faint)]'"
      >
        <span class="text-base">{{ item.icon }}</span>
        <span>{{ item.label }}</span>
        <span
          v-if="$route.path === item.path"
          ref="indicatorRef"
          class="absolute bottom-0 left-2 right-2 h-0.5 rounded-full"
          style="background: var(--clay)"
        ></span>
      </router-link>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import { gsap } from '../../composables/useGSAP'

const route = useRoute()
const navRef = ref<HTMLElement>()
const navLinksRef = ref<HTMLElement>()
const indicatorRef = ref<HTMLElement>()
const linkRefs = ref<HTMLElement[]>([])

const navItems = [
  { path: '/', label: '对话', icon: '◈' },
  { path: '/knowledge', label: '知识库', icon: '⧉' },
  { path: '/evaluation', label: '评估', icon: '⧗' },
]

let ctx: gsap.Context | null = null

onMounted(() => {
  if (!navRef.value) return
  ctx = gsap.context(() => {
    // Initial slide-in
    gsap.from(navRef.value, {
      y: -48,
      duration: 0.5,
      ease: 'power3.out',
      delay: 0.3,
    })
  }, navRef.value)
})

onUnmounted(() => {
  ctx?.revert()
})

// Animate active indicator on route change
watch(() => route.path, async () => {
  await nextTick()
  const activeLink = linkRefs.value.find((el) => {
    const link = el?.querySelector('a')
    return link?.classList.contains('active-link')
  })
  if (!activeLink) return
  const indicator = activeLink.querySelector('[ref="indicatorRef"]') as HTMLElement
  if (!indicator) return
  gsap.fromTo(indicator,
    { scaleX: 0.6, opacity: 0 },
    { scaleX: 1, opacity: 1, duration: 0.3, ease: 'power2.inOut' }
  )
})
</script>

<style scoped>
.active-link {
  background: var(--clay-wash);
  color: var(--clay);
}
</style>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/layout/TopNav.vue
git commit -m "feat: add TopNav slide-in and active indicator GSAP animations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: ChatMessage print impression animation

**Files:**
- Modify: `frontend/src/components/chat/ChatMessage.vue`

- [ ] **Step 1: Add print impression message enter animation**

Replace the component with GSAP-driven animations:

```vue
<template>
  <article ref="msgRef" class="py-6" :class="message.role === 'user' ? 'user-msg' : 'assistant-msg'">
    <!-- Horizontal rule (separator) -->
    <div ref="ruleRef" class="msg-rule mb-3" :class="message.role === 'user' ? 'rule-user' : 'rule-assistant'"></div>

    <!-- Label -->
    <div class="flex items-center gap-2.5 mb-3">
      <span ref="dotRef" class="label-dot" :class="message.role === 'user' ? 'dot-user' : 'dot-assistant'"></span>
      <span ref="labelRef" class="text-sm font-semibold uppercase tracking-widest" style="font-family: var(--font-body)"
        :style="{ color: message.role === 'user' ? 'var(--ink-muted)' : 'var(--clay)' }">
        {{ message.role === 'user' ? 'You' : 'RAG Agent' }}
      </span>
      <span class="text-xs tracking-wide opacity-30">{{ message.role === 'assistant' && message.sourceType ? routeLabel : '' }}</span>
    </div>

    <!-- Content with gradient mask reveal -->
    <div ref="contentRef" class="content-reveal text-lg leading-relaxed" style="color: var(--ink)" v-html="renderedContent"></div>

    <!-- Sources -->
    <div v-if="message.role === 'assistant' && message.sources?.length" class="mt-4">
      <SourcePanel :sources="message.sources" />
    </div>
  </article>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import type { ChatMessage } from '../../types'
import SourcePanel from './SourcePanel.vue'
import { gsap } from '../../composables/useGSAP'

const props = defineProps<{ message: ChatMessage }>()
const msgRef = ref<HTMLElement>()
const ruleRef = ref<HTMLElement>()
const dotRef = ref<HTMLElement>()
const labelRef = ref<HTMLElement>()
const contentRef = ref<HTMLElement>()

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

    // 1. Horizontal rule expands from center
    tl.fromTo(ruleRef.value,
      { scaleX: 0, opacity: 0 },
      { scaleX: 1, opacity: 1, duration: 0.25, ease: 'power3.out' }
    )

    // 2. Label stamp effect
    tl.fromTo(labelRef.value,
      { scale: 0.8, rotation: props.message.role === 'user' ? 2 : -2, opacity: 0 },
      { scale: 1, rotation: 0, opacity: 1, duration: 0.3, ease: 'back.out(1.3)' },
      '+=0.05'
    )

    // 3. Dot pop
    tl.fromTo(dotRef.value,
      { scale: 0 },
      { scale: 1, duration: 0.2, ease: 'back.out(1.4)' },
      '<'
    )

    // 4. Content gradient mask reveal
    tl.fromTo(contentRef.value,
      { '--mask-pos': '0%' },
      { '--mask-pos': '100%', duration: 0.5, ease: 'power3.inOut' },
      '+=0.05'
    )

    // 5. Paper texture shadow (subtle)
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

/* Horizontal rule */
.msg-rule {
  height: 1px;
  transform-origin: center;
}
.rule-assistant { background: var(--rule); }
.rule-user { background: var(--rule-light); }

/* Gradient mask reveal */
.content-reveal {
  --mask-pos: 0%;
  -webkit-mask-image: linear-gradient(to right, black var(--mask-pos), transparent var(--mask-pos));
  mask-image: linear-gradient(to right, black var(--mask-pos), transparent var(--mask-pos));
}
</style>
```

Note: The gradient mask approach using `--mask-pos` CSS custom property doesn't work as a simple GSAP tween (GSAP can't animate custom properties used in `mask-image` directly). Replace with a `clip-path` approach:

Actually, for the content reveal, a simpler and more robust approach is to use `clip-path`:

```typescript
// In the timeline, replace the content reveal step with:
tl.fromTo(contentRef.value,
  { clipPath: 'inset(0 100% 0 0)', opacity: 0.8 },
  { clipPath: 'inset(0 0% 0 0)', opacity: 1, duration: 0.5, ease: 'power3.inOut' },
  '+=0.05'
)
```

And remove the `--mask-pos` CSS custom property approach.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/chat/ChatMessage.vue
git commit -m "feat: add print impression message enter animation

Horizontal rule expansion, label stamp, dot pop, and clip-path
content reveal for the 'freshly printed document' metaphor.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: AgentThinking wave dot animation

**Files:**
- Modify: `frontend/src/components/chat/AgentThinking.vue`

- [ ] **Step 1: Replace CSS keyframes with GSAP timeline loop**

```vue
<template>
  <div v-if="isStreaming" ref="thinkRef" class="flex items-center gap-3 py-4">
    <div ref="dotsRef" class="flex items-center gap-1">
      <span class="think-dot" ref="dot1"></span>
      <span class="think-dot" ref="dot2"></span>
      <span class="think-dot" ref="dot3"></span>
    </div>
    <span ref="labelRef" class="text-xs tracking-wide italic" style="color: var(--ink-muted)">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { gsap } from '../../composables/useGSAP'

const props = defineProps<{ route: string; isStreaming: boolean }>()

const thinkRef = ref<HTMLElement>()
const dotsRef = ref<HTMLElement>()
const dot1 = ref<HTMLElement>()
const dot2 = ref<HTMLElement>()
const dot3 = ref<HTMLElement>()
const labelRef = ref<HTMLElement>()

const label = computed(() => {
  const m: Record<string, string> = {
    'web': 'Searching the web…',
    'rag': 'Reading your documents…',
    'direct': 'Reasoning…',
  }
  return m[props.route] || 'Thinking…'
})

let ctx: gsap.Context | null = null

onMounted(() => {
  if (!thinkRef.value) return
  ctx = gsap.context(() => {
    const dots = [dot1.value, dot2.value, dot3.value]

    // Wave dot animation — each dot pulses in sequence
    const dotTL = gsap.timeline({ repeat: -1 })
    dots.forEach((dot, i) => {
      dotTL.to(dot, {
        scale: 1,
        opacity: 1,
        backgroundColor: '#c2653d', // var(--clay)
        duration: 0.3,
        ease: 'power2.out',
      }, i * 0.15)
      dotTL.to(dot, {
        scale: 0.4,
        opacity: 0.2,
        backgroundColor: '#c4bfb8', // var(--ink-subtle)
        duration: 0.3,
        ease: 'power2.in',
      }, '+=0.3')
    })

    // Label pulse
    gsap.to(labelRef.value, {
      opacity: 0.6,
      duration: 0.9,
      repeat: -1,
      yoyo: true,
      ease: 'sine.inOut',
    })
  }, thinkRef.value)
})

onUnmounted(() => {
  ctx?.revert()
})
</script>

<style scoped>
.think-dot {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--ink-subtle);
  transform: scale(0.4);
  opacity: 0.2;
}
/* Remove old @keyframes think animation */
</style>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/chat/AgentThinking.vue
git commit -m "feat: replace CSS dot animation with GSAP wave dot loop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: ChatInput send button feedback

**Files:**
- Modify: `frontend/src/components/chat/ChatInput.vue`

- [ ] **Step 1: Add GSAP send button click feedback**

Modify the `submit` function and add refs:

```vue
<template>
  <form @submit.prevent="submit" class="relative"
    style="background: var(--white); border: 1px solid var(--rule); border-radius: var(--r-xl); box-shadow: 0 1px 3px rgba(0,0,0,0.03)">
    <textarea
      ref="inputEl"
      v-model="input"
      rows="1"
      placeholder="Ask anything…"
      class="w-full rounded-xl px-5 py-4 text-lg outline-none resize-none"
      style="background: transparent; color: var(--ink); min-height: 60px; max-height: 200px; line-height: 1.5"
      :disabled="disabled"
      @keydown.enter.exact.prevent="submit"
      @keydown.enter.shift.exact="input += '\n'"
      @input="autoResize"
      @focus="focused = true"
      @blur="focused = false"
    ></textarea>
    <div class="flex items-center justify-between px-4 pb-3">
      <span class="text-[10px] tracking-wider opacity-0 transition-opacity"
        :style="{ opacity: focused ? 0.4 : 0 }">Enter to send · Shift+Enter for new line</span>
      <button ref="btnRef" type="submit" :disabled="disabled || !input.trim()"
        class="ml-auto px-4 py-1.5 text-sm font-medium rounded-lg transition-colors duration-200 disabled:opacity-30"
        style="background: var(--clay); color: white">
        <span v-if="!disabled">Send</span>
        <span v-else class="flex items-center gap-1.5">
          <span class="inline-block w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
          Thinking
        </span>
      </button>
    </div>
  </form>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { gsap } from '../../composables/useGSAP'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ send: [text: string] }>()

const input = ref('')
const focused = ref(false)
const inputEl = ref<HTMLTextAreaElement>()
const btnRef = ref<HTMLElement>()

onMounted(() => inputEl.value?.focus())

function autoResize() {
  const el = inputEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 160) + 'px'
}

function submit() {
  if (!input.value.trim() || props.disabled) return

  // GSAP send button feedback
  if (btnRef.value) {
    gsap.timeline()
      .to(btnRef.value, {
        backgroundColor: 'var(--clay-bright)',
        duration: 0.15,
        ease: 'power2.out',
      })
      .to(btnRef.value, {
        backgroundColor: 'var(--clay)',
        duration: 0.3,
        ease: 'power2.out',
      }, '+=0.05')
      .to(btnRef.value, {
        scale: 0.95,
        duration: 0.1,
        ease: 'power2.in',
      }, '<')
      .to(btnRef.value, {
        scale: 1,
        duration: 0.25,
        ease: 'back.out(1.3)',
      })
  }

  emit('send', input.value.trim())
  input.value = ''
  const el = inputEl.value
  if (el) el.style.height = 'auto'
}
</script>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/chat/ChatInput.vue
git commit -m "feat: add GSAP send button click feedback animation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: SourcePanel stagger reveal

**Files:**
- Modify: `frontend/src/components/chat/SourcePanel.vue`

- [ ] **Step 1: Add staggered source item reveal**

First, read the current SourcePanel.vue, then add GSAP:

```vue
<template>
  <div ref="panelRef" class="source-panel">
    <div
      v-for="(source, i) in sources" :key="i" :ref="el => sourceRefs[i] = el"
      class="source-item text-xs leading-relaxed py-2 px-3 rounded-lg"
      style="background: var(--white-warm); border: 1px solid var(--rule-light)"
    >
      <p class="font-semibold mb-0.5" style="color: var(--clay); font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em">
        {{ source.title || 'Source ' + (i + 1) }}
      </p>
      <p class="line-clamp-2" style="color: var(--ink-soft)">{{ source.snippet || source.text }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { gsap } from '../../composables/useGSAP'

interface Source {
  title?: string
  snippet?: string
  text?: string
}

defineProps<{ sources: Source[] }>()

const panelRef = ref<HTMLElement>()
const sourceRefs = ref<(HTMLElement | null)[]>([])

let ctx: gsap.Context | null = null

onMounted(() => {
  if (!panelRef.value) return
  ctx = gsap.context(() => {
    // Filter out null refs
    const items = sourceRefs.value.filter(Boolean) as HTMLElement[]
    gsap.from(items, {
      opacity: 0,
      y: 6,
      stagger: 0.05,
      duration: 0.3,
      ease: 'power2.out',
    })
  }, panelRef.value)
})

onUnmounted(() => {
  ctx?.revert()
})
</script>
```

Note: You need to first read the actual `SourcePanel.vue` to see its current template structure and adapt accordingly.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/chat/SourcePanel.vue
git commit -m "feat: add staggered source item GSAP reveal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: ConversationSidebar animations

**Files:**
- Modify: `frontend/src/components/chat/ConversationSidebar.vue`

- [ ] **Step 1: Add panel enter/exit, item hover, and delete animations**

First read the current file, then add GSAP. Key animations:

```typescript
// Panel enter animation (run when sidebar becomes visible)
function animatePanelEnter() {
  if (!sidebarRef.value) return
  ctx = gsap.context(() => {
    const tl = gsap.timeline()
    tl.from(sidebarRef.value, {
      x: -280,
      opacity: 0,
      duration: 0.4,
      ease: 'power3.out',
    })
    tl.from('.sidebar-title', {
      opacity: 0,
      y: -8,
      duration: 0.3,
    }, '+=0.05')
    tl.from('.conversation-item', {
      opacity: 0,
      x: -20,
      stagger: 0.04,
      duration: 0.3,
      ease: 'power2.out',
    }, '+=0.05')
  }, sidebarRef.value)
}

// Delete animation
function animateDelete(itemEl: HTMLElement, onComplete: () => void) {
  gsap.timeline({
    onComplete,
  })
    .to(itemEl, {
      height: 0,
      opacity: 0,
      x: -30,
      duration: 0.25,
      ease: 'power3.in',
    })
    .to(itemEl, {
      backgroundColor: 'var(--clay-wash)',
      duration: 0.15,
    }, 0)
}

// Item hover (use CSS for simple hover, but GSAP for the action button reveal)
// In template: @mouseenter / @mouseleave
function onItemEnter(el: HTMLElement) {
  gsap.to(el, { x: 4, duration: 0.2, ease: 'power2.out' })
  // Reveal action buttons
  const btn = el.querySelector('.action-btn') as HTMLElement
  if (btn) gsap.to(btn, { opacity: 1, scale: 1, duration: 0.2, ease: 'back.out(1.2)' })
}
function onItemLeave(el: HTMLElement) {
  gsap.to(el, { x: 0, duration: 0.2, ease: 'power2.out' })
  const btn = el.querySelector('.action-btn') as HTMLElement
  if (btn) gsap.to(btn, { opacity: 0, scale: 0.8, duration: 0.15, ease: 'power2.in' })
}
```

Note: The actual implementation details depend on the current ConversationSidebar.vue template structure. Read the file first and adapt the selectors accordingly.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/chat/ConversationSidebar.vue
git commit -m "feat: add sidebar panel, hover, and delete GSAP animations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: ChatView animations (empty state, smooth scroll, streaming cursor, context panel)

**Files:**
- Modify: `frontend/src/views/ChatView.vue`

- [ ] **Step 1: Add empty state, smooth scroll, and streaming cursor animations**

Replace the relevant parts:

```typescript
// Smooth scroll (replace the watch → scrollTop behavior)
watch(messages, async () => {
  await nextTick()
  if (msgContainer.value) {
    gsap.to(msgContainer.value, {
      scrollTop: msgContainer.value.scrollHeight,
      duration: 0.5,
      ease: 'power2.out',
    })
  }
}, { deep: true })

watch(streamingContent, async () => {
  await nextTick()
  if (msgContainer.value) {
    gsap.to(msgContainer.value, {
      scrollTop: msgContainer.value.scrollHeight,
      duration: 0.3,
      ease: 'power2.out',
    })
  }
})

// Streaming cursor — replace the CSS blink with GSAP
// In onMounted:
onMounted(() => {
  fetchConversations()

  // Animate empty state if present
  const emptyTitle = document.querySelector('.empty-title') as HTMLElement
  if (emptyTitle) {
    gsap.to(emptyTitle, {
      y: -2,
      duration: 2,
      repeat: -1,
      yoyo: true,
      ease: 'sine.inOut',
    })
  }

  // Streaming cursor pulse
  const cursor = document.querySelector('.cursor-blink') as HTMLElement
  if (cursor) {
    gsap.to(cursor, {
      opacity: 0,
      duration: 0.4,
      repeat: -1,
      yoyo: true,
      ease: 'sine.inOut',
    })
    gsap.to(cursor, {
      scaleX: 0.6,
      duration: 0.4,
      repeat: -1,
      yoyo: true,
      ease: 'sine.inOut',
    })
  }
})
```

Note: The empty state animation and cursor animation need to be created inside `gsap.context()` scoped to the view. Read the actual ChatView.vue first to adapt the selectors.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/views/ChatView.vue
git commit -m "feat: add smooth scroll, empty state, and cursor GSAP animations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: FileUploader border breathe animation

**Files:**
- Modify: `frontend/src/components/knowledge/FileUploader.vue`

- [ ] **Step 1: Add file drop feedback animation**

Read the current file first. Add GSAP for the file-accepted feedback:

```typescript
function animateFileAccepted() {
  if (!uploaderRef.value) return
  gsap.timeline()
    // Border breathe
    .to(uploaderRef.value, {
      borderColor: 'var(--clay)',
      boxShadow: '0 0 0 8px var(--clay-glow)',
      duration: 0.3,
      ease: 'power2.out',
    })
    .to(uploaderRef.value, {
      borderColor: 'var(--rule)',
      boxShadow: '0 0 0 0px transparent',
      duration: 0.3,
      ease: 'power2.in',
    })
    // Filename pop-in
    .from('.filename-label', {
      opacity: 0,
      y: -12,
      scale: 0.9,
      duration: 0.35,
      ease: 'back.out(1.3)',
    }, 0.1)
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/knowledge/FileUploader.vue
git commit -m "feat: add file drop border breathe GSAP animation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: KnowledgeView card enter + ScrollTrigger

**Files:**
- Modify: `frontend/src/views/KnowledgeView.vue`

- [ ] **Step 1: Add card enter and ScrollTrigger reveal**

```typescript
import { gsap, ScrollTrigger } from '../composables/useGSAP'

onMounted(() => {
  fetchDocuments()

  if (!containerRef.value) return
  ctx = gsap.context(() => {
    // Animate existing document cards on mount
    const cards = document.querySelectorAll('.doc-card')
    if (cards.length) {
      gsap.from(cards, {
        opacity: 0,
        x: -16,
        rotateY: 8,
        duration: 0.4,
        stagger: 0.08,
        ease: 'power3.out',
      })
      // Also animate the type badge inside each card
      gsap.from('.doc-badge', {
        scale: 0,
        rotation: -90,
        duration: 0.3,
        stagger: 0.08,
        ease: 'back.out(1.4)',
      })
    }

    // ScrollTrigger: cards that come into view later get revealed
    ScrollTrigger.batch('.doc-card', {
      onEnter: (elements) => {
        gsap.fromTo(elements,
          { opacity: 0, y: 30, rotateX: 8 },
          { opacity: 1, y: 0, rotateX: 0, duration: 0.5, stagger: 0.08, ease: 'power3.out' }
        )
      },
      start: 'top 85%',
      once: true,
    })
  }, containerRef.value)
})
```

Note: Add `ref="containerRef"` to the root div in the template, and add `class="doc-card"` to each document card element, and `class="doc-badge"` to the type badge.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/views/KnowledgeView.vue
git commit -m "feat: add document card enter and ScrollTrigger GSAP reveal

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: EvaluationView result animations (print effect, progress bars, countUp)

**Files:**
- Modify: `frontend/src/views/EvaluationView.vue`

- [ ] **Step 1: Add result card print effect and progress bar animations**

```typescript
import { gsap, ScrollTrigger } from '../composables/useGSAP'

// After results load, animate them in
watch(results, async (newResults) => {
  if (!newResults.length) return
  await nextTick()

  const resultCards = document.querySelectorAll('.result-card')
  resultCards.forEach((card, i) => {
    const tl = gsap.timeline({ delay: i * 0.1 })

    // Separator line
    const rule = card.querySelector('.result-rule') as HTMLElement
    if (rule) {
      tl.fromTo(rule, { scaleX: 0 }, { scaleX: 1, duration: 0.25, ease: 'power3.out' })
    }

    // "Q:" stamp
    const qLabel = card.querySelector('.q-label') as HTMLElement
    if (qLabel) {
      tl.fromTo(qLabel,
        { scale: 0.8, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.2, ease: 'back.out(1.3)' },
        '+=0.05'
      )
    }

    // Query text clip-path reveal
    const qText = card.querySelector('.q-text') as HTMLElement
    if (qText) {
      tl.fromTo(qText,
        { clipPath: 'inset(0 100% 0 0)' },
        { clipPath: 'inset(0 0% 0 0)', duration: 0.4, ease: 'power3.inOut' },
        '+=0.03'
      )
    }

    // "A:" stamp + answer reveal (same pattern)
    // ... similar for .a-label and .a-text

    // Progress bars — animate width and countUp
    const bar = card.querySelector('.progress-bar-fill') as HTMLElement
    const scoreEl = card.querySelector('.score-value') as HTMLElement
    if (bar && scoreEl) {
      const targetWidth = parseFloat(bar.dataset.target || '0')
      const targetScore = parseFloat(scoreEl.dataset.score || '0')

      tl.to(bar, {
        width: targetWidth + '%',
        duration: 0.7,
        ease: 'power3.out',
      }, '+=0.05')

      // CountUp effect
      tl.to({ val: 0 }, {
        val: targetScore,
        duration: 0.7,
        ease: 'power2.out',
        onUpdate: function() {
          if (scoreEl) scoreEl.textContent = Math.round(this.targets()[0].val) + '%'
        },
      }, '<0.2')

      // Flash if score >= 80
      if (targetScore >= 80) {
        tl.to(scoreEl, {
          scale: 1.15,
          duration: 0.15,
          ease: 'power2.out',
        })
        tl.to(scoreEl, {
          scale: 1,
          duration: 0.15,
          ease: 'power2.in',
        })
      }
    }
  })
})
```

Note: You need to add CSS classes (`.result-card`, `.result-rule`, `.q-label`, `.q-text`, `.a-label`, `.a-text`, `.progress-bar-fill`, `.score-value`) and `data-target` / `data-score` attributes to the template elements. Read the current EvaluationView.vue first to adapt selectors to the actual template.

- [ ] **Step 2: Commit**

```bash
git add frontend/src/views/EvaluationView.vue
git commit -m "feat: add evaluation result GSAP animations with countUp

Result card print impression reveal, progress bar fill animation,
number countUp effect, and score flash for high scores.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 13: Final verification and cleanup

**Files:** None (verification only)

- [ ] **Step 1: Build check**

```bash
cd frontend && npm run build
```
Expected: Build succeeds with no TypeScript or Vite errors.

- [ ] **Step 2: Visual verification**

```bash
cd frontend && npm run dev
```
Expected: App loads. Navigate between routes, send messages, check:
- Route transitions are smooth GSAP animations (not CSS)
- TopNav slides in on first load
- Messages appear with print impression effect
- Agent thinking dots animate in wave pattern
- Send button has click feedback
- Knowledge cards stagger in, animate on scroll
- Evaluation progress bars count up

- [ ] **Step 3: Reduced motion check**

In browser DevTools, enable `prefers-reduced-motion: reduce`. Verify animations are skipped (instant transitions).

- [ ] **Step 4: Commit final cleanup**

```bash
git add .
git commit -m "feat: final cleanup and verification for GSAP animations

All 14 animation items implemented across 11 components.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Easing Reference

| Context | Easing |
|---|---|
| Entrance reveals | `power3.out` |
| Exit / collapse | `power2.in` |
| Bounce / pop | `back.out(1.2–1.4)` |
| Looping / pulse | `sine.inOut` |
| Progress bar fill | `power3.out` |
| CountUp | `power2.out` |
| Stagger per item | `power2.out` |

## GSAP Patterns Used

- **gsap.context(scope)** — all animations scoped to component root; auto-cleanup on unmount via `ctx.revert()`
- **gsap.timeline()** — sequencing multi-step animations (message enter, result cards)
- **gsap.from() / gsap.fromTo()** — entrance reveals
- **gsap.matchMedia()** — global reduced-motion support
- **ScrollTrigger.batch()** — batched scroll-driven card reveals
- **gsap.registerPlugin(ScrollTrigger)** — once at app init
