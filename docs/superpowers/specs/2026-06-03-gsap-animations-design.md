# GSAP Frontend Animations — Design Spec

**Date:** 2026-06-03
**Project:** RAG 智能问答机器人 (Agentic RAG)
**Stack:** Vue 3 + TypeScript + Tailwind CSS v4 + GSAP
**Design System:** Editorial Research Desk — warm paper, terracotta accent, print-quality typography

---

## 1. Overview

Add a rich, cohesive animation language to the frontend using GSAP. The "Print Impression" metaphor guides message animations; page elements follow a staggered narrative-reveal pattern. Everything respects `prefers-reduced-motion`.

### Skills Used

| Skill | Purpose |
|---|---|
| `gsap-core` | `gsap.to/from/fromTo`, easing, stagger, defaults, `matchMedia` |
| `gsap-timeline` | Sequencing, position parameter |
| `gsap-scrolltrigger` | Scroll-driven reveals, smooth scrolling |
| `gsap-frameworks` | Vue 3 lifecycle (`useGSAP`, `gsap.context`, cleanup on unmount) |

---

## 2. Animation Inventory

### 2.1 Route Transition (App.vue)

Replace existing CSS `<transition>` with a GSAP timeline driven by `useGSAP` + router hooks.

```
Timeline (routeChange):
├─ [0.0s] 旧页面     to: opacity 0, y -8px, scale 0.995   (0.2s, power2.in)
├─ [0.2s] 新页面     from: opacity 0, y 12px              (0.35s, power3.out)
└─ [0.2s] 新页面元素  pageEnter sequence (see 2.2)
```

### 2.2 Page Enter Sequence (All Views)

```
Timeline (pageEnter), delay: 0.1s from route change:
├─ [0.0s] h1 标题          from: opacity 0, y 24px → to: 1,0      (0.6s, power3.out)
├─ [0.1s] 副标题/描述 p     from: opacity 0, y 16px → to: 1,0      (0.5s, power3.out)
├─ [0.2s] 主要内容区        stagger 0.08s, from: opacity 0, y 20px (0.5s, power2.out)
└─ [0.4s] 次要元素          from: opacity 0, y 12px                (0.4s, power2.out)
```

View-specific content areas:
- **ChatView:** empty-state text → ChatInput
- **KnowledgeView:** FileUploader → document cards
- **EvaluationView:** query card → result items

### 2.3 TopNav (TopNav.vue)

```
Initial load:
  nav container  from: y -48px → to: y 0      (0.5s, power3.out, delay 0.3s)

Active indicator:
  Highlight underline slides between nav items on click (Flip-like, 0.3s power2.inOut)
```

### 2.4 Message Enter — Print Impression (ChatMessage.vue)

Assistant messages use a "freshly printed document" metaphor:

```
Timeline (messageEnter):
├─ [0.00s] 水平分隔线       width: 0→100% (center origin)         (0.25s, power3.out)
├─ [0.10s] "RAG Agent" 标签 scale: 0.8→1.05→1, rotate ±2°         (0.3s, back.out(1.3))
├─ [0.20s] 内容文字         渐变遮罩从左→右扫过揭示 (gradient mask sweep)
│                            clip-path: inset(0 100% 0 0) → inset(0 0 0 0)
│                            (0.5s, power3.inOut)
│                            注: 内容通过 v-html 渲染，用遮罩而非逐字动画
├─ [0.30s] 来源面板         height: 0→auto, opacity 0→1           (0.3s, power2.out)
└─ [0.50s] 底部阴影         微弱的"撕纸"纹理阴影出现              (0.2s)
```

User messages are similar but mirrored direction (separator from right, label stamp from right side).

Source panel (SourcePanel.vue) — nested inside message:
```
stagger 0.05s per source item, from: opacity 0, y 6px
```

### 2.5 Agent Thinking Indicator (AgentThinking.vue)

Replace static 3-dot CSS animation with GSAP timeline loop:

```
Timeline (thinking, repeating):
├─ Dot 1  scale: 0.4→1→0.4, background: ink-subtle → clay → ink-subtle  (0.6s/step, stagger 0.15s)
├─ Dot 2  (same, offset 0.15s)
├─ Dot 3  (same, offset 0.3s)
└─ 标签文字 opacity pulse: 0.6→1→0.6 (1.8s loop, sine.inOut)

Route-specific labels:
  web → "Searching the web…"
  rag → "Reading your documents…"
  direct → "Reasoning…"
  fallback → "Thinking…"
```

### 2.6 Streaming Cursor

Replace CSS `@keyframes blink` with GSAP pulse:

```
gsap.to(cursorEl, {
  opacity: 0, duration: 0.4, repeat: -1, yoyo: true,
  ease: "sine.inOut"
})
// Also slight scaleX oscillation: 1 → 0.6 → 1
```

### 2.7 Send Button (ChatInput.vue)

```
Click feedback:
├─ Arrow icon  rotate: 0→-30°→0, x: 0→4px→0  (0.3s, power2.inOut)
├─ Background  var(--clay) → var(--clay-bright) → var(--clay)  (0.3s)
└─ On disabled/spinner: three-data-point scanning loop
```

### 2.8 Conversation Sidebar (ConversationSidebar.vue)

```
Panel open:
├─ Container     from: x -280px, opacity 0 → to: 0, 1  (0.4s, power3.out)
├─ Title         from: opacity 0, y -8px                (0.3s, delay 0.05s)
└─ Items         stagger 0.04s, from: opacity 0, x -20  (0.3s, power2.out)

Item hover:
├─ Highlight     background sweep left→right, like a highlighter stroke (0.35s)
├─ Text          x: 0→4px (0.2s)
└─ Action btn    from: opacity 0, scale 0.8  (0.2s, back.out(1.2))

Item delete:
├─ [0.0s] Height auto→0 (0.3s, power3.in)
├─ [0.0s] Opacity 1→0, x 0→-30px (0.25s)
├─ [0.0s] Flash var(--clay-wash) bg (0.15s)
└─ [0.3s] Remaining items shift up to fill gap (0.25s, power2.out)

New chat button:
├─ Hover: "+" rotate 0→90° (0.25s, power2.out)
└─ Border: var(--rule)→var(--clay)
```

### 2.9 Empty State (ChatView)

```
Timeline (emptyState, looping):
├─ Title words     y drift ±2px per line (4s loop, sine.inOut)
├─ Italic "need"   skewX: 0→3°→0→-3°→0 (6s loop)
└─ Decorative line under "need": slow horizontal sweep
```

### 2.10 Knowledge Base — File Upload (FileUploader.vue)

```
File dropped / selected:
├─ [0.0s] Border "breathe": var(--rule) → var(--clay) → var(--rule) (0.6s)
│          boxShadow: none → 0 0 0 8px var(--clay-glow) → none
├─ [0.1s] Filename label pop-in: from y -12px, scale 0.9 → to 0,1   (0.35s, back.out(1.3))
└─ [完成]  Card inserted at top of list with stagger
```

### 2.11 Knowledge Base — Document Cards

```
Card enter (stagger 0.08s):
├─ Card          from: opacity 0, x -16px, rotateY 8° → 0,0,0   (0.4s, power3.out)
├─ Type badge    from: scale 0, rotate -90° → 1,0               (0.3s, back.out(1.4))
└─ Filename+meta from: opacity 0, x -8px → 1,0                  (0.25s, power2.out)

Card hover:
├─ Lift y: 0→-3px (0.25s)
├─ boxShadow deepen (0.25s)
├─ Type badge bg invert: wash→clay (0.2s)
└─ Left edge 2px clay indicator appears (opacity 0→1)
```

### 2.12 Knowledge Base — ScrollTrigger Reveal

Applied to the document card grid. Note: when cards are already visible in the viewport on page load, ScrollTrigger handles the reveal; the pageEnter sequence (2.2) skips the card area and only animates FileUploader. This avoids double-animating the same elements.

```
ScrollTrigger (document cards section):
├─ trigger: .grid container
├─ start: "top 85%"
├─ Per card: from opacity 0, y 30px, rotateX 8° → 1,0,0 (0.5s, power3.out)
├─ stagger: 0.08s
├─ scrub: false
├─ once: true
```

### 2.13 Evaluation — Run Button

```
Click → Running state:
├─ Text fade out → spinner in (3 data-dot scanning loop)
├─ Pulse ring around button (one-shot scale+opacity decay)
└─ Disabled state: opacity 0.3
```

### 2.14 Evaluation — Result Cards (Print Effect Variant)

```
Timeline (resultCard, stagger 0.1s per card):
├─ [0.0s]  Separator line expand from center (0.25s, power3.out)
├─ [0.1s]  "Q:" stamp label  (0.2s, back.out(1.3))
├─ [0.15s] Query text        gradient mask sweep reveal  (0.4s, power3.inOut)
├─ [0.3s]  "A:" stamp label  (0.2s, back.out(1.3))
├─ [0.35s] Answer text       gradient mask sweep reveal  (0.4s, power3.inOut)
└─ [0.5s]  Progress bars     0→target% (0.7s, power3.out)
```

### 2.15 Evaluation — Progress Bars & CountUp

```
Timeline (scoreReveal, per card):
├─ [0.00s] Bar width: 0 → target%                      (0.7s, power3.out)
│          Color: var(--rule) → green/red based on score
├─ [0.15s] Number countUp: 0 → target                   (0.7s, power2.out)
│          onUpdate: display = Math.round(v) + '%'
└─ [0.40s] If score ≥ 80%: number scale 1→1.15→1 flash (0.15s)
```

ScrollTrigger for bars: only animate when visible (`start: "top 80%"`, `once: true`).

### 2.16 Evaluation — ScrollTrigger Bars

```
ScrollTrigger (per result card):
├─ trigger: .result-card
├─ start: "top 80%"
├─ Bars animate on enter
├─ once: true
```

### 2.17 Smooth Scroll (ChatView)

Replace native `scrollTop` with GSAP smooth scroll when new messages arrive:

```ts
gsap.to(msgContainer, {
  scrollTop: msgContainer.scrollHeight,
  duration: 0.5,
  ease: "power2.out",
})
```

### 2.18 Context Panel (ChatView)

```
Timeline (panelToggle):
├─ Panel       from: x 40px, opacity 0 → 0, 1           (0.35s, power3.out)
├─ Main area   x: 0 → -10px (micro-shift)                (0.35s, power3.out, concurrent)
└─ Source items stagger 0.06s, from: x 12px, opacity 0   (0.25s, power2.out)
```

### 2.19 Reduced Motion (Global)

```ts
// In main.ts or a dedicated composable
gsap.matchMedia().add("(prefers-reduced-motion: no-preference)", () => {
  gsap.defaults({})
})

gsap.matchMedia().add("(prefers-reduced-motion: reduce)", () => {
  gsap.defaults({ duration: 0, overwrite: true })
})
```

---

## 3. Implementation Structure

### 3.1 New Files

| File | Purpose |
|---|---|
| `src/composables/useGSAP.ts` | Shared composable: imports GSAP defaults, matchMedia setup, `useGSAP` re-export |
| `src/composables/usePageEnter.ts` | Page enter timeline factory, shared across views |

### 3.2 Modified Files

| File | Changes |
|---|---|
| `package.json` | Add `gsap` dependency |
| `src/main.ts` | Import GSAP, call matchMedia setup |
| `src/App.vue` | Replace CSS transition with GSAP route transition |
| `src/views/ChatView.vue` | Message enter, empty state, smooth scroll, context panel |
| `src/views/KnowledgeView.vue` | Card enter, ScrollTrigger reveal |
| `src/views/EvaluationView.vue` | Result card enter, progress bars, countUp, ScrollTrigger |
| `src/components/layout/TopNav.vue` | Nav slide-in, active indicator slide |
| `src/components/chat/ChatMessage.vue` | Print impression enter |
| `src/components/chat/ChatInput.vue` | Send button feedback |
| `src/components/chat/AgentThinking.vue` | Wave dot loop, label pulse |
| `src/components/chat/ConversationSidebar.vue` | Panel enter/exit, hover, delete |
| `src/components/chat/SourcePanel.vue` | Source item stagger |
| `src/components/knowledge/FileUploader.vue` | Drop feedback, border breathe |
| `src/components/knowledge/DocList.vue` | Card hover (or handled in KnowledgeView) |

### 3.3 Easing Conventions

| Context | Easing |
|---|---|
| Entrance reveals | `power3.out` |
| Exit / collapse | `power2.in` |
| Bounce / pop | `back.out(1.2–1.4)` |
| Looping / pulse | `sine.inOut` |
| Scroll | `power2.out` |
| Stagger | `power2.out` per item |

### 3.4 Vue Lifecycle Pattern

Every component using GSAP:

```vue
<script setup lang="ts">
import { useGSAP } from '../composables/useGSAP'
import gsap from 'gsap'

const container = ref<HTMLElement>()

useGSAP(() => {
  // All tweens/timelines inside gsap.context auto-scoped
  // Auto-cleanup on unmount
  const tl = gsap.timeline()
  tl.from('.title', { opacity: 0, y: 24, duration: 0.6, ease: 'power3.out' })
  // ...
}, { scope: container })
</script>
```

---

## 4. Constraints

- **No breaking changes** to existing layout or functionality
- **Tailwind v4 compatibility** — GSAP animates inline `style` properties; Tailwind classes remain the source of truth for static styles
- **Performance** — all transforms use `x`/`y`/`scale`/`rotate` (GPU-composited), no `left`/`top`/`margin` animations
- **Accessibility** — `prefers-reduced-motion` honored globally via `gsap.matchMedia()`
- **ScrollTrigger** — all scroll-driven animations fire only once (`once: true`)
- **Incremental delivery** — each component can be animated independently; no cross-component animation dependencies

---

## 5. Out of Scope

- Page-level particle/background animations (decorative)
- Dark mode (project does not currently support it)
- Sound effects
- 3D transforms beyond subtle rotateX/Y for cards
