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

export interface ConversationSummary {
  id: number
  title: string
  message_count: number
  last_message_preview: string | null
  created_at: string | null
  updated_at: string | null
}

export interface ConversationDetail {
  conversation: {
    id: number
    title: string
    created_at: string | null
    updated_at: string | null
  }
  summary: string | null
  messages: {
    id: number
    role: string
    content: string
    source_type: string | null
    created_at: string | null
  }[]
}
