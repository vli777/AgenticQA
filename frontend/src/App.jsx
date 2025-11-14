import { useMemo, useRef, useState } from 'react'
import './App.css'

const DEFAULT_NAMESPACE = 'default'
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

const formatScore = (score) => {
  if (typeof score !== 'number') return null
  return (score * 100).toFixed(1)
}

function App() {
  const [namespace, setNamespace] = useState(DEFAULT_NAMESPACE)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [uploadSummary, setUploadSummary] = useState(null)
  const [uploadError, setUploadError] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [chatMode, setChatMode] = useState('agentic')
  const [isSending, setIsSending] = useState(false)
  const [chatError, setChatError] = useState('')
  const [isClearing, setIsClearing] = useState(false)
  const [clearMessage, setClearMessage] = useState('')
  const [clearError, setClearError] = useState('')

  const fileInputRef = useRef(null)

  const sanitizedNamespace = useMemo(() => {
    const trimmed = namespace.trim()
    return trimmed.length > 0 ? trimmed : DEFAULT_NAMESPACE
  }, [namespace])

  const handleFileSelection = (event) => {
    const files = Array.from(event.target.files || [])
    setSelectedFiles(files)
    setUploadError('')
    setUploadSummary(null)
    setClearMessage('')
    setClearError('')
  }

  const handleNamespaceBlur = () => {
    if (!namespace.trim()) {
      setNamespace(DEFAULT_NAMESPACE)
    }
  }

  const handleClearNamespace = async () => {
    if (isClearing) return
    const ns = sanitizedNamespace
    const confirmed = window.confirm(`Delete all indexed vectors in namespace "${ns}"? This cannot be undone.`)
    if (!confirmed) return

    setIsClearing(true)
    setClearMessage('')
    setClearError('')

    try {
      const response = await fetch(`${API_BASE_URL}/debug/namespace/${encodeURIComponent(ns)}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail?.detail || `Failed to clear namespace (status ${response.status})`)
      }

      const data = await response.json().catch(() => ({}))
      const clearedNamespace = data?.namespace ?? ns
      if (data?.warning || data?.vectors_deleted === false) {
        setClearMessage(`Namespace "${clearedNamespace}" already empty. Cache cleared for good measure.`)
      } else {
        setClearMessage(`Namespace "${clearedNamespace}" cleared.`)
      }
      setUploadSummary(null)
      setMessages([])
    } catch (error) {
      const fallbackMessage = `Namespace "${ns}" cleared (best effort).`
      if (error?.message?.toLowerCase().includes('failed to fetch')) {
        console.warn('Clear namespace request failed, assuming already empty.', error)
        setClearMessage(fallbackMessage)
      } else {
        setClearError(error?.message || 'Unexpected error while clearing namespace.')
      }
    } finally {
      setIsClearing(false)
    }
  }

  const handleUpload = async () => {
    if (!selectedFiles.length) {
      setUploadError('Select at least one PDF or TXT file before uploading.')
      return
    }

    const url = new URL(`${API_BASE_URL}/upload/`)
    url.searchParams.set('namespace', sanitizedNamespace)

    const formData = new FormData()
    selectedFiles.forEach((file) => formData.append('files', file))

    setIsUploading(true)
    setUploadError('')

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail?.detail || `Upload failed with status ${response.status}`)
      }

      const data = await response.json()
      setUploadSummary({
        indexedChunks: data?.indexed_chunks ?? 0,
        fileCount: selectedFiles.length,
      })
      setSelectedFiles([])
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    } catch (error) {
      setUploadError(error.message || 'Unexpected error while uploading files.')
    } finally {
      setIsUploading(false)
    }
  }

  const appendMessage = (message) => {
    setMessages((prev) => [...prev, message])
  }

  const handleSend = async (event) => {
    event.preventDefault()
    const question = inputValue.trim()
    if (!question) return

    appendMessage({ role: 'user', content: question })
    setInputValue('')
    setChatError('')
    setIsSending(true)

    const endpoint = chatMode === 'agentic' ? '/ask/agentic' : '/ask/'
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, namespace: sanitizedNamespace }),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail?.detail || `Request failed with status ${response.status}`)
      }

      const data = await response.json()

      if (chatMode === 'agentic') {
        appendMessage({
          role: 'assistant',
          mode: 'agentic',
          content: data?.answer || 'No answer returned.',
          reasoning: Array.isArray(data?.reasoning) ? data.reasoning : [],
          sources: Array.isArray(data?.sources) ? data.sources : [],
        })
      } else {
        const matches = Array.isArray(data?.results?.matches) ? data.results.matches : []
        appendMessage({
          role: 'assistant',
          mode: 'rag',
          content: matches.length ? 'Here is what I found:' : 'No high-confidence matches were found.',
          matches: matches.map((match, index) => ({
            id: match.id ?? `match-${index}`,
            score: match.score,
            text: match.metadata?.text || '',
            source: match.metadata?.source || match.metadata?.doc_id || 'Unknown source',
          })),
        })
      }
    } catch (error) {
      setChatError(error.message || 'Unexpected error while contacting the backend.')
      appendMessage({
        role: 'assistant',
        mode: 'error',
        content: 'Sorry, I could not process that request.',
      })
    } finally {
      setIsSending(false)
    }
  }

  const renderAssistantMessage = (message) => {
    if (message.mode === 'agentic') {
      return (
        <div className="agentic-response">
          <p className="answer-text">{message.content}</p>
          {message.reasoning?.length ? (
            <div className="reasoning-block">
              <h4>Reasoning</h4>
              <ol>
                {message.reasoning.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ol>
            </div>
          ) : null}
          {message.sources?.length ? (
            <div className="sources-block">
              <h4>Sources</h4>
              <ul>
                {message.sources.map((source, index) => (
                  <li key={index}>{source}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      )
    }

    if (message.mode === 'rag') {
      return (
        <div className="rag-response">
          <p className="answer-text">{message.content}</p>
          {message.matches?.length ? (
            <ul className="match-list">
              {message.matches.map((match) => (
                <li key={match.id}>
                  <div className="match-source">{match.source}</div>
                  {match.score !== undefined ? (
                    <div className="match-score">{formatScore(match.score)}% match</div>
                  ) : null}
                  <p>{match.text}</p>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      )
    }

    return <p className="answer-text">{message.content}</p>
  }

  return (
    <div className="app-shell">
      <aside className="side-panel">
        <header>
          <h1>AgenticQA</h1>
          <p className="tagline">Upload documents, then ask questions via standard RAG or the agentic workflow.</p>
        </header>

        <section className="namespace-section">
          <div className="namespace-header">
            <label htmlFor="namespace">Namespace</label>
            <button
              type="button"
              className="clear-namespace"
              onClick={handleClearNamespace}
              disabled={isClearing}
            >
              {isClearing ? 'Clearing…' : 'Clear'}
            </button>
          </div>
          <input
            id="namespace"
            type="text"
            value={namespace}
            onChange={(event) => setNamespace(event.target.value)}
            onBlur={handleNamespaceBlur}
            placeholder="default"
            autoComplete="off"
          />
          <p className="field-hint">Use namespaces to keep document sets separate. Leave as "default" for a single corpus.</p>
          {clearMessage ? <p className="success-message namespace-message">{clearMessage}</p> : null}
          {clearError ? <p className="error-message namespace-message">{clearError}</p> : null}
        </section>

        <section className="upload-section">
          <h2>Document Upload</h2>
          <p className="section-description">Supported formats: PDF, TXT. Files are chunked and indexed into Pinecone.</p>

          <label className="upload-control" htmlFor="file-input">
            <input
              id="file-input"
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt"
              onChange={handleFileSelection}
            />
          </label>

          <button type="button" onClick={handleUpload} disabled={isUploading}>
            {isUploading ? 'Uploading…' : 'Upload & Index'}
          </button>

          {uploadSummary ? (
            <div className="upload-summary">
              <strong>Upload complete.</strong>
              <p>
                {uploadSummary.fileCount} file{uploadSummary.fileCount === 1 ? '' : 's'} processed ·{' '}
                {uploadSummary.indexedChunks} chunk{uploadSummary.indexedChunks === 1 ? '' : 's'} indexed
              </p>
            </div>
          ) : null}

          {clearMessage ? <p className="success-message">{clearMessage}</p> : null}
          {uploadError ? <p className="error-message">{uploadError}</p> : null}
          {clearError ? <p className="error-message">{clearError}</p> : null}
        </section>
      </aside>

      <main className="chat-panel">
        <div className="chat-header">
          <div>
            <h2>Chat</h2>
            <p className="section-description">Ask the assistant about the documents you have indexed.</p>
          </div>
          <div className="mode-switcher">
            <button
              type="button"
              className={chatMode === 'rag' ? 'active' : ''}
              onClick={() => setChatMode('rag')}
            >
              RAG
            </button>
            <button
              type="button"
              className={chatMode === 'agentic' ? 'active' : ''}
              onClick={() => setChatMode('agentic')}
            >
              Agentic
            </button>
          </div>
        </div>

        <div className="message-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              <p>Start by uploading documents, then ask a question to see answers here.</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-meta">
                  <span className="speaker">{message.role === 'user' ? 'You' : 'Assistant'}</span>
                  {message.mode === 'agentic' ? <span className="mode-chip">Agentic</span> : null}
                  {message.mode === 'rag' ? <span className="mode-chip">RAG</span> : null}
                  {message.mode === 'error' ? <span className="mode-chip error">Error</span> : null}
                </div>
                <div className="message-content">
                  {message.role === 'assistant' ? renderAssistantMessage(message) : (
                    <p className="answer-text">{message.content}</p>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        <form className="message-composer" onSubmit={handleSend}>
          <textarea
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder={chatMode === 'agentic' ? 'Ask a complex question – the agent can plan multiple searches…' : 'Ask a question about your documents…'}
            rows={chatMode === 'agentic' ? 3 : 2}
            disabled={isSending}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                if (isSending || !inputValue.trim()) {
                  return
                }
                const target = event.target
                requestAnimationFrame(() => {
                  target.form?.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }))
                })
              }
            }}
          />
          <div className="composer-actions">
            <span className="composer-hint">Shift + Enter for new line</span>
            <button type="submit" disabled={isSending || !inputValue.trim()}>
              {isSending ? 'Thinking…' : 'Send'}
            </button>
          </div>
        </form>

        {chatError ? <p className="error-message chat-error">{chatError}</p> : null}
      </main>
    </div>
  )
}

export default App
