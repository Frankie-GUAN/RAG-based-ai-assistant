<template>
  <aside class="sidebar" :class="{ collapsed: !visible }">
    <!-- New Chat button -->
    <div class="new-chat-container">
      <button @click="$emit('newChat')" class="new-chat-btn">
        <span class="plus-icon">+</span>
        <span>New Chat</span>
      </button>
    </div>

    <!-- Conversation list -->
    <div class="conv-list">
      <!-- Loading skeleton -->
      <div v-if="loading" class="space-y-2">
        <div v-for="n in 3" :key="n" class="skeleton-item">
          <div class="skeleton-line w-3/4"></div>
          <div class="skeleton-line w-1/2 mt-1.5"></div>
        </div>
      </div>

      <!-- Empty state -->
      <div v-else-if="conversations.length === 0" class="empty-state">
        <p class="empty-text">No conversations yet</p>
      </div>

      <!-- List -->
      <div v-else class="space-y-0.5">
        <div
          v-for="conv in conversations"
          :key="conv.id"
          class="conv-item"
          :class="{ active: conv.id === activeId }"
          @click="onSelectConv(conv.id)"
        >
          <div v-if="editingId === conv.id" class="flex gap-1.5 items-center" @click.stop>
            <input
              ref="editInput"
              v-model="editTitle"
              class="edit-input"
              maxlength="100"
              @keydown.enter="saveRename(conv.id)"
              @keydown.escape="cancelRename"
              @blur="saveRename(conv.id)"
            />
          </div>
          <div v-else class="conv-title" @dblclick.stop="startRename(conv.id, conv.title)">
            {{ conv.title }}
          </div>
          <div class="conv-preview">{{ conv.last_message_preview }}</div>
          <div class="conv-meta">
            <span>{{ formatDate(conv.updated_at) }}</span>
            <span>{{ conv.message_count }} msgs</span>
          </div>
          <button
            class="delete-btn"
            @click.stop="$emit('delete', conv.id)"
            title="Delete conversation"
          >&times;</button>
        </div>
      </div>
    </div>

    <!-- Toggle -->
    <button class="toggle-btn" @click="visible = !visible" :title="visible ? 'Collapse' : 'Expand'">
      {{ visible ? '◀' : '▶' }}
    </button>
  </aside>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue'
import type { ConversationSummary } from '../../types'

defineProps<{
  conversations: ConversationSummary[]
  activeId: number | null
  loading: boolean
}>()

const emit = defineEmits<{
  select: [id: number]
  delete: [id: number]
  newChat: []
  rename: [id: number, title: string]
}>()

const visible = ref(true)
const editingId = ref<number | null>(null)
const editTitle = ref('')
const editInput = ref<HTMLInputElement>()

function onSelectConv(id: number) {
  if (editingId.value !== id) {
    emit('select', id)
  }
}

async function startRename(id: number, title: string) {
  editingId.value = id
  editTitle.value = title
  await nextTick()
  editInput.value?.focus()
  editInput.value?.select()
}

function saveRename(id: number) {
  if (editingId.value !== id) return
  const newTitle = editTitle.value.trim()
  if (newTitle && newTitle.length > 0) {
    emit('rename', id, newTitle)
  }
  editingId.value = null
  editTitle.value = ''
}

function cancelRename() {
  editingId.value = null
  editTitle.value = ''
}

function formatDate(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  return d.toLocaleDateString('zh-CN')
}
</script>

<style scoped>
.sidebar {
  width: 260px;
  min-width: 260px;
  display: flex;
  flex-direction: column;
  background: var(--white-warm);
  border-right: 1px solid var(--rule);
  transition: width 200ms ease, min-width 200ms ease;
  position: relative;
}
.sidebar.collapsed {
  width: 0;
  min-width: 0;
  overflow: hidden;
  border-right: none;
}

.new-chat-container {
  padding: 16px 14px 10px;
}

.new-chat-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 0;
  font-size: 15px;
  font-weight: 600;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  background: var(--clay);
  color: white;
  transition: opacity 150ms;
}
.new-chat-btn:hover { opacity: 0.88; }

.plus-icon {
  font-size: 18px;
  font-weight: 400;
  line-height: 1;
}

.conv-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 10px 10px;
}

.conv-item {
  display: block;
  width: 100%;
  text-align: left;
  padding: 12px 12px 10px;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  background: transparent;
  position: relative;
  transition: background 120ms;
}
.conv-item:hover { background: var(--rule-faint); }
.conv-item.active { background: var(--clay-wash); }

.conv-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 24px;
}

.conv-preview {
  font-size: 12px;
  color: var(--ink-muted);
  margin-top: 3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.conv-meta {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--ink-subtle);
  margin-top: 5px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.delete-btn {
  position: absolute;
  top: 10px;
  right: 8px;
  width: 20px;
  height: 20px;
  border: none;
  background: transparent;
  color: var(--ink-muted);
  font-size: 16px;
  cursor: pointer;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 120ms;
}
.conv-item:hover .delete-btn { opacity: 0.5; }
.delete-btn:hover { opacity: 1 !important; color: #e53e3e; }

.empty-state {
  padding: 32px 16px;
  text-align: center;
}
.empty-text {
  font-size: 13px;
  color: var(--ink-muted);
}

.skeleton-item {
  padding: 12px;
  border-radius: 10px;
  background: var(--rule-faint);
}
.skeleton-line {
  height: 10px;
  border-radius: 4px;
  background: var(--rule);
  animation: pulse 1.5s infinite;
}
.skeleton-line.w-3\/4 { width: 75%; }
.skeleton-line.w-1\/2 { width: 50%; }

@keyframes pulse {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 0.7; }
}

.toggle-btn {
  position: absolute;
  top: 50%;
  right: -18px;
  transform: translateY(-50%);
  width: 18px;
  height: 36px;
  border: 1px solid var(--rule);
  border-left: none;
  border-radius: 0 6px 6px 0;
  background: var(--white-warm);
  color: var(--ink-muted);
  font-size: 10px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  opacity: 0;
  transition: opacity 120ms;
}
.sidebar:hover .toggle-btn { opacity: 0.6; }

.edit-input {
  width: 100%;
  padding: 3px 6px;
  font-size: 13px;
  font-weight: 600;
  color: var(--ink);
  background: var(--white);
  border: 1px solid var(--clay);
  border-radius: 6px;
  outline: none;
}
.edit-input:focus {
  border-color: var(--clay);
  box-shadow: 0 0 0 2px var(--clay-wash);
}
</style>
