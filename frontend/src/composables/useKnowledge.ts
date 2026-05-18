import { ref } from 'vue'
import type { DocumentRecord } from '../types'

export function useKnowledge() {
  const documents = ref<DocumentRecord[]>([])
  const isUploading = ref(false)

  async function upload(file: File) {
    isUploading.value = true
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch('/api/knowledge/upload', {
      method: 'POST',
      body: formData,
    })
    const result = await response.json()
    await fetchDocuments()
    isUploading.value = false
    return result
  }

  async function fetchDocuments() {
    const response = await fetch('/api/knowledge/documents')
    documents.value = await response.json()
  }

  return { documents, isUploading, upload, fetchDocuments }
}
