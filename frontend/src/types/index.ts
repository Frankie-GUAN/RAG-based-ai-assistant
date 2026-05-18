export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  sourceType?: 'web' | 'doc'
}

export interface Source {
  title: string
  link: string
  snippet?: string
}

export interface DocumentRecord {
  id: number
  filename: string
  file_type: string
  chunk_count: number
  created_at: string
}

export interface EvalResult {
  query: string
  answer: string
  faithfulness: number
  answer_relevancy: number
}
