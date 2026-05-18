import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { DocumentRecord } from '../types'

export const useKnowledgeStore = defineStore('knowledge', () => {
  const documents = ref<DocumentRecord[]>([])

  function setDocuments(docs: DocumentRecord[]) {
    documents.value = docs
  }

  return { documents, setDocuments }
})
