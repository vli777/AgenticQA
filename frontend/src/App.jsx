import { useMemo, useRef, useState } from 'react'
import './App.css'

const DEFAULT_NAMESPACE = 'default'
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

function App() {
  const [namespace, setNamespace] = useState(DEFAULT_NAMESPACE)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [uploadSummary, setUploadSummary] = useState(null)
  const [uploadError, setUploadError] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0, fileName: '', status: '', step: 0, totalSteps: 0 })
  const [messages, setMessages] = useState([])
  const [queuedQueries, setQueuedQueries] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [chatError, setChatError] = useState('')
  const [streamingStatus, setStreamingStatus] = useState('')
  const [isClearing, setIsClearing] = useState(false)
  const [clearMessage, setClearMessage] = useState('')
  const [clearError, setClearError] = useState('')
  const conversationId = useMemo(() => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID()
    }
    return `conv-${Date.now()}-${Math.random().toString(36).slice(2)}`
  }, [])

  const fileInputRef = useRef(null)

  const sanitizedNamespace = useMemo(() => {
    const trimmed = namespace.trim()
    return trimmed.length > 0 ? trimmed : DEFAULT_NAMESPACE
  }, [namespace])

  const handleFileSelection = (event) => {
    const file = event.target.files?.[0]
    setSelectedFiles(file ? [file] : [])
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
      setUploadError('Please select a PDF or TXT file to upload.')
      return
    }

    const url = new URL(`${API_BASE_URL}/upload/`)
    url.searchParams.set('namespace', sanitizedNamespace)

    const formData = new FormData()
    selectedFiles.forEach((file) => formData.append('files', file))

    setIsUploading(true)
    setUploadError('')
    setUploadProgress({ current: 0, total: selectedFiles.length, fileName: '', status: 'Starting upload...', step: 0, totalSteps: 6 })

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        throw new Error(detail?.detail || `Upload failed with status ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let totalChunks = 0

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue

          try {
            const data = JSON.parse(line.slice(6))

            if (data.type === 'progress') {
              setUploadProgress({
                current: data.current,
                total: data.total,
                fileName: data.file_name,
                status: data.status,
                step: data.step || 0,
                totalSteps: data.total_steps || 6
              })
            } else if (data.type === 'complete') {
              totalChunks = data.indexed_chunks
            } else if (data.type === 'error') {
              throw new Error(data.message || 'Upload failed')
            }
          } catch (parseError) {
            console.error('Failed to parse SSE data:', parseError)
          }
        }
      }

      setUploadSummary({
        indexedChunks: totalChunks,
      })
      setSelectedFiles([])
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
      setUploadProgress({ current: 0, total: 0, fileName: '', status: '', step: 0, totalSteps: 0 })
    } catch (error) {
      setUploadError(error.message || 'Unexpected error while uploading files.')
      setUploadProgress({ current: 0, total: 0, fileName: '', status: '', step: 0, totalSteps: 0 })
    } finally {
      setIsUploading(false)
    }

    // Process any queued queries now that upload is complete
    // Must be after isUploading is set to false
    await processQueuedQueries()
  }

  const appendMessage = (message) => {
    setMessages((prev) => [...prev, message])
  }

  const sendQuery = async (question) => {
    setChatError('')
    setIsSending(true)
    setStreamingStatus('Connecting...')

    // Build URL with query parameters for GET request
    const url = new URL(`${API_BASE_URL}/ask`)
    url.searchParams.set('question', question)
    url.searchParams.set('namespace', sanitizedNamespace)
    url.searchParams.set('conversation_id', conversationId)

    let eventSource = null
    let currentAnswer = ''
    // Calculate message index when we start - user message was already added
    const currentMessageIndex = messages.length

    try {
      eventSource = new EventSource(url.toString())

      eventSource.onopen = () => {
        setStreamingStatus('Connected')
      }

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)

          if (data.type === 'reasoning') {
            // Update streaming status with reasoning step
            setStreamingStatus(data.content)
          } else if (data.type === 'answer') {
            // Append answer content
            currentAnswer = data.content

            // Update or append the assistant message
            setMessages((prev) => {
              const newMessages = [...prev]
              if (newMessages[currentMessageIndex] && newMessages[currentMessageIndex].role === 'assistant') {
                // Update existing message
                newMessages[currentMessageIndex] = {
                  ...newMessages[currentMessageIndex],
                  content: currentAnswer,
                }
              } else {
                // Add new message
                newMessages.push({
                  role: 'assistant',
                  mode: 'agentic',
                  content: currentAnswer,
                })
              }
              return newMessages
            })
          } else if (data.type === 'done') {
            // Stream complete
            setStreamingStatus('')
            setIsSending(false)
            eventSource.close()
          } else if (data.type === 'error') {
            setChatError(data.content || 'Streaming error occurred')
            setStreamingStatus('')
            setIsSending(false)
            eventSource.close()
          }
        } catch (parseError) {
          console.error('Failed to parse SSE data:', parseError)
        }
      }

      eventSource.onerror = (error) => {
        console.error('EventSource error:', error)
        setChatError('Connection to server lost')
        setStreamingStatus('')
        setIsSending(false)

        if (eventSource) {
          eventSource.close()
        }

        // Add error message if no answer was received
        if (!currentAnswer) {
          appendMessage({
            role: 'assistant',
            mode: 'error',
            content: 'Sorry, I could not process that request.',
          })
        }
      }
    } catch (error) {
      setChatError(error.message || 'Unexpected error while contacting the backend.')
      setStreamingStatus('')
      setIsSending(false)

      appendMessage({
        role: 'assistant',
        mode: 'error',
        content: 'Sorry, I could not process that request.',
      })
    }
  }

  const handleSend = async (event) => {
    event.preventDefault()
    const question = inputValue.trim()
    if (!question) return

    // If uploading, queue the query instead of sending immediately
    if (isUploading) {
      setQueuedQueries(prev => {
        const newQueue = [...prev, question]
        setStreamingStatus(`Query queued (${newQueue.length} in queue). Will process after upload completes...`)
        return newQueue
      })
      appendMessage({ role: 'user', content: question })
      setInputValue('')
      return
    }

    appendMessage({ role: 'user', content: question })
    setInputValue('')
    await sendQuery(question)
  }

  const processQueuedQueries = async () => {
    if (queuedQueries.length === 0) {
      setStreamingStatus('')
      return
    }

    console.log('Processing queued queries:', queuedQueries.length)
    const queries = [...queuedQueries]
    setQueuedQueries([]) // Clear queue immediately

    // Batch queries intelligently based on estimated token count
    // Rough estimate: 4 chars per token, max ~2000 tokens per batch for safety
    // This leaves plenty of room for RAG context, system prompts, and response
    const MAX_CHARS_PER_BATCH = 8000
    const batches = []
    let currentBatch = []
    let currentBatchSize = 0

    for (const query of queries) {
      const querySize = query.length
      const overhead = 50 // "Please answer the following questions:\n\n1. " etc.

      if (currentBatchSize + querySize + overhead > MAX_CHARS_PER_BATCH && currentBatch.length > 0) {
        // Start a new batch
        batches.push([...currentBatch])
        currentBatch = [query]
        currentBatchSize = querySize + overhead
      } else {
        currentBatch.push(query)
        currentBatchSize += querySize + overhead
      }
    }

    if (currentBatch.length > 0) {
      batches.push(currentBatch)
    }

    // Process each batch
    console.log('Total batches to process:', batches.length)
    try {
      for (let i = 0; i < batches.length; i++) {
        const batch = batches[i]
        const batchInfo = batches.length > 1 ? ` (batch ${i + 1}/${batches.length})` : ''
        setStreamingStatus(`Processing ${batch.length} queued ${batch.length === 1 ? 'query' : 'queries'}${batchInfo}...`)

        console.log(`Processing batch ${i + 1}/${batches.length}:`, batch)

        if (batch.length === 1) {
          await sendQuery(batch[0])
        } else {
          const combinedQuery = batch.map((q, idx) => `${idx + 1}. ${q}`).join('\n\n')
          const wrappedQuery = `Please answer the following questions:\n\n${combinedQuery}`
          console.log('Combined query:', wrappedQuery)
          await sendQuery(wrappedQuery)
        }

        // Small delay between batches
        if (i < batches.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 500))
        }
      }
    } catch (error) {
      console.error('Error processing queued queries:', error)
      setChatError(`Failed to process queued queries: ${error.message}`)
    } finally {
      setStreamingStatus('')
    }
  }

  const renderAssistantMessage = (message) => {
    if (message.mode === 'error') {
      return <p className="answer-text">{message.content}</p>
    }

    return (
      <div className="agentic-response">
        <p className="answer-text">{message.content}</p>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <aside className="side-panel">
        <header>
          <h1>AgenticQA</h1>
          <p className="tagline">Upload documents, then ask questions about them.</p>
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
          <p className="section-description">Upload one document at a time. Supported formats: PDF, DOCX, TXT.</p>

          <label className="upload-control" htmlFor="file-input">
            <input
              id="file-input"
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={handleFileSelection}
            />
          </label>

          <button type="button" onClick={handleUpload} disabled={isUploading}>
            {isUploading ? 'Uploading…' : 'Upload'}
          </button>

          {isUploading && (
            <div className="upload-progress">
              <div className="progress-info">
                {uploadProgress.step > 0 && (
                  <strong>Step {uploadProgress.step} of {uploadProgress.totalSteps}</strong>
                )}
                {uploadProgress.status && <p className="progress-status">{uploadProgress.status}</p>}
              </div>
              {uploadProgress.totalSteps > 0 && (
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${(uploadProgress.step / uploadProgress.totalSteps) * 100}%` }}
                  ></div>
                </div>
              )}
            </div>
          )}

          {uploadSummary ? (
            <div className="upload-summary">
              <strong>Upload complete.</strong>
              <p>
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

        {streamingStatus && (
          <div className="streaming-status">
            <span className="status-indicator"></span>
            {streamingStatus}
          </div>
        )}

        <form className="message-composer" onSubmit={handleSend}>
          <textarea
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder="Ask a question about your documents…"
            rows={3}
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
            <span className="composer-hint">
              {!uploadSummary ? 'Upload documents first' : 'Shift + Enter for new line'}
            </span>
            <button type="submit" disabled={isSending || !inputValue.trim() || !uploadSummary}>
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
