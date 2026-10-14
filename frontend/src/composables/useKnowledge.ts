import { ref } from 'vue'
import type { DocumentRecord } from '../types'

export function useKnowledge() {
  const documents = ref<DocumentRecord[]>([])
  const isUploading = ref(false)
  /** 删除中的文档 id 集合：让每张卡片各自转圈，而不是整页转圈 */
  const deletingIds = ref<Set<number>>(new Set())
  const error = ref('')

  async function upload(file: File) {
    isUploading.value = true
    error.value = ''
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/knowledge/upload', {
        method: 'POST',
        body: formData,
      })
      const result = await response.json()

      if (!response.ok) {
        // 413（超限）/ 422（解析不出内容）/ 400 都是后端给出的可读原因，
        // 直接透出它，不要把它们变成一句无信息量的「上传失败」
        error.value = result.detail || `上传失败（HTTP ${response.status}）`
        return null
      }

      await fetchDocuments()
      return result
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      isUploading.value = false
    }
  }

  async function fetchDocuments() {
    const response = await fetch('/api/knowledge/documents')
    documents.value = await response.json()
  }

  async function deleteDocument(id: number) {
    error.value = ''
    deletingIds.value = new Set(deletingIds.value).add(id)
    try {
      const response = await fetch(`/api/knowledge/documents/${id}`, { method: 'DELETE' })
      if (!response.ok) {
        const result = await response.json().catch(() => ({}))
        error.value = result.detail || `删除失败（HTTP ${response.status}）`
        return false
      }
      await fetchDocuments()
      return true
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
      return false
    } finally {
      const next = new Set(deletingIds.value)
      next.delete(id)
      deletingIds.value = next
    }
  }

  return { documents, isUploading, deletingIds, error, upload, fetchDocuments, deleteDocument }
}
