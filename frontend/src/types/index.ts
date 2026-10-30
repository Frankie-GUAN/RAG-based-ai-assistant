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
  /** 文件名（不含章节）—— 章节信息拼接进 title 后，仍需要原名做展示与去重 */
  sourceName?: string
  /** 章节路径，如 "Pointer Analysis: Foundations › Worklist (WL)" */
  section?: string
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
  /** null = 判官没能给出可解析的分数。不再把它伪装成 0（那会伪造出一条质量问题） */
  faithfulness: number | null
  answer_relevancy: number | null
  error?: string | null
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
    /** 引用溯源：检索命中的片段（后端 source_data 解析而来） */
    sources?: Array<{
      doc_id: string
      chunk: string
      source: string
      score: number
    }>
    created_at: string | null
  }[]
}
