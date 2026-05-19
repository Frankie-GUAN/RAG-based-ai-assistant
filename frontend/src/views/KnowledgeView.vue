<template>
  <div class="h-full overflow-y-auto">
    <div class="max-w-4xl mx-auto px-8 py-12">
      <!-- Page title -->
      <h1 class="text-3xl mb-2" style="font-family: var(--font-display)">Knowledge Base</h1>
      <p class="mb-10" style="color: var(--ink-muted)">Upload documents to build your private search index. The Agent retrieves from here when answers need document grounding.</p>

      <!-- Upload -->
      <FileUploader :is-uploading="isUploading" @upload="handleUpload" />

      <!-- Document list — 2-column grid -->
      <section class="mt-12">
        <h2 class="text-xs uppercase tracking-widest mb-6" style="color: var(--ink-muted); font-family: var(--font-body)">
          {{ documents.length }} Document{{ documents.length !== 1 ? 's' : '' }} indexed
        </h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div v-for="doc in documents" :key="doc.id"
            class="flex items-center gap-4 p-5 rounded-xl transition-shadow duration-200 hover:shadow-sm"
            style="background: var(--white); border: 1px solid var(--rule)">
            <div class="w-10 h-10 rounded-lg flex items-center justify-center text-sm shrink-0"
              style="background: var(--clay-wash); color: var(--clay)">
              {{ doc.file_type.toUpperCase() }}
            </div>
            <div class="min-w-0 flex-1">
              <p class="text-sm font-semibold truncate">{{ doc.filename }}</p>
              <p class="text-xs mt-0.5" style="color: var(--ink-muted)">{{ doc.chunk_count }} chunks · {{ formatDate(doc.created_at) }}</p>
            </div>
          </div>
        </div>
        <div v-if="!documents.length" class="text-center py-20" style="color: var(--ink-faint)">
          <p class="text-4xl mb-4 opacity-20">⧉</p>
          <p class="text-sm">No documents yet. Upload a PDF or TXT file above.</p>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useKnowledge } from '../composables/useKnowledge'
import FileUploader from '../components/knowledge/FileUploader.vue'

const { documents, isUploading, upload, fetchDocuments } = useKnowledge()
onMounted(() => fetchDocuments())
async function handleUpload(file: File) { await upload(file) }
function formatDate(iso: string) { return new Date(iso).toLocaleDateString('zh-CN') }
</script>
